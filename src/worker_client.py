"""
Harness-side handle on the tool worker.

Modes (config `execution.mode`):
  subprocess (default)  worker = `python -m src.worker_main` in its own session.
                        A crash / OOM kill / hang in a tool kills only the worker.
  container             worker runs in a docker/podman container (default image:
                        the lab runtime image, ghcr.io/vanallenlab/agentic-sc-runtime:v2):
                        separate memory/CPU limits (cgroups),
                        read-only inputs and code, writable output dir only,
                        optional network isolation.
  inprocess             old behaviour: tools run inside the harness process
                        (debugging only; a crash kills the run).

When the worker dies or exceeds the hard timeout, the harness kills it, starts a
new one from the latest per-step checkpoints (output/checkpoints/) and reports
the failure to the model as the result of the call that caused it.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from src.harness_guard import DATA_SLOTS
from src.logging_utils import log, logger
from src.worker_protocol import WorkerDied, WorkerTimeout, read_frame, write_frame

# Lab runtime image (conda env on PATH); see onboarding/README.md
DEFAULT_IMAGE = "ghcr.io/vanallenlab/agentic-sc-runtime:v2"

DEFAULT_ENV_PASSTHROUGH = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
    "SYNAPSE_AUTH_TOKEN", "NCBI_API_KEY", "ENTREZ_EMAIL", "SERPAPI_API_KEY",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
)


# LLM credentials are never placed in the worker's environment: agent-written
# code (REPL / terminal) must not be able to call an LLM, e.g. to have another
# model look at a plot. Harness tools that need them (literature search) get
# them in memory via the init request; see ToolRuntime._tool_env.
LLM_SECRET_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")


class WorkerFatal(Exception):
    """The worker could not be (re)started; the run cannot continue."""


def _describe_exit(code: Optional[int]) -> str:
    if code is None:
        return "did not respond (killed by the harness)"
    if code in (-signal.SIGKILL, 137):
        return f"was killed (exit {code}; SIGKILL — typically out of memory)"
    if code in (-signal.SIGSEGV, 139):
        return f"crashed with a segmentation fault (exit {code})"
    if code in (-signal.SIGABRT, 134):
        return f"aborted (exit {code}; e.g. a Rust/C++ panic in a native extension)"
    if code is not None and code < 0:
        return f"was terminated by signal {signal.Signals(-code).name}"
    return f"exited unexpectedly (exit code {code})"


class _BaseClient:
    def __init__(self, init_payload: dict, output_dir: Path, tool_timeout_s=None):
        self.init_payload = dict(init_payload)
        self.output_dir = Path(output_dir)
        self.tool_timeout_s = tool_timeout_s
        self.generation = 0
        self.info: dict = {}
        self.alive = False

    def _checkpoint_paths(self) -> dict:
        ckpt_dir = self.output_dir / "checkpoints"
        found = {}
        for attr in DATA_SLOTS:
            for ext in (".h5ad", ".csv"):
                p = ckpt_dir / f"{attr}{ext}"
                if p.exists():
                    found[attr] = str(p)
        return found


class InProcessClient(_BaseClient):
    """Runs the ToolRuntime in the harness process (no isolation)."""

    def start(self):
        from src.worker_runtime import ToolRuntime
        self.runtime = ToolRuntime()
        self.info = self.runtime.init(dict(self.init_payload, generation=0))["info"]
        self.alive = True
        return self.info

    def call(self, name, args, exec_id):
        return self.runtime.call({"name": name, "args": args, "exec_id": exec_id})

    def checkpoint(self):
        return self.runtime.checkpoint()["results"]

    def finalize(self):
        return self.runtime.finalize()

    def shutdown(self):
        self.alive = False


def _backed_component_paths(path: str) -> list[str]:
    """Per-sample files a SnapATAC2 AnnDataSet (backed scATAC object) points to.

    They are opened by path when the dataset is read, so they must be mounted
    alongside the dataset file itself."""
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "uns/AnnDataSet/file_path" not in f:
                return []
            rel = [x.decode() if isinstance(x, bytes) else str(x) for x in f["uns/AnnDataSet/file_path"][:]]
    except Exception as e:  # noqa: BLE001 - not an AnnDataSet / unreadable: nothing extra to mount
        log(f"⚠️ could not read component paths of {path}: {e}")
        return []
    base = Path(path).parent
    return [str((base / p).resolve()) if not os.path.isabs(p) else str(Path(p).resolve()) for p in rel]


_BLOCKED_HOSTS = ("169.254.169.254", "metadata.google.internal", "metadata.goog")


def _blacklist_terms(blacklist) -> list[str]:
    """Lowercase substrings that mark a result as coming from a blacklisted source."""
    if not blacklist:
        return []
    terms = set()
    for key in ("dois", "urls", "title_contains", "pmcids", "pmcid"):
        for v in blacklist.get(key) or []:
            v = str(v).strip().lower()
            for prefix in ("https://", "http://", "www."):
                if v.startswith(prefix):
                    v = v[len(prefix):]
            if len(v) >= 8:  # never match on short fragments
                terms.add(v)
    for v in blacklist.get("pmids") or []:
        v = str(v).strip()
        if len(v) >= 7:
            terms.add(v)
    return sorted(terms, key=len, reverse=True)


def _first_blacklist_hit(text: str, terms: list[str]):
    if not text or not terms:
        return None
    low = text.lower()
    for t in terms:
        if t in low:
            return t
    return None


class WorkerClient(_BaseClient):
    def __init__(self, init_payload: dict, output_dir: Path, repo_dir: Path, mode: str = "subprocess",
                 tool_timeout_s=None, hard_timeout_grace_s: float = 120.0, container: dict = None,
                 memory_limit_gb=None, agent_code_llm_keys: bool = False,
                 harness_tools: frozenset = frozenset()):
        super().__init__(init_payload, output_dir, tool_timeout_s)
        self.agent_code_llm_keys = bool(agent_code_llm_keys)
        self.repo_dir = Path(repo_dir).resolve()
        self.mode = mode
        self.grace = float(hard_timeout_grace_s)
        self.container = container or {}
        # container mode: "code" mounts only <repo>/src (no configs, tasks, outputs, data or .git);
        # "repo" mounts the whole repository read-only (old behaviour)
        self.mount = self.container.get("mount", "code")
        if self.mount not in ("code", "repo"):
            raise ValueError(f"execution.container.mount must be code | repo, got {self.mount!r}")
        self.memory_limit_gb = memory_limit_gb
        # Tools executed in the harness process instead of the worker (network tools when the
        # worker has no network). They never run agent-written code and never touch the data.
        self.harness_tools = frozenset(harness_tools)
        self._harness_runtime = None
        self.proc: Optional[subprocess.Popen] = None
        self.container_name: Optional[str] = None
        self._busy = False

    # ------------------------------------------------------------- process
    def _command(self) -> list[str]:
        worker_script = str(self.repo_dir / "src" / "worker_main.py")
        if self.mode == "subprocess":
            return [sys.executable, worker_script]
        c = self.container
        runtime = c.get("runtime", "docker")
        self.container_name = f"m3a-worker-{os.getpid()}-{self.generation}"
        cmd = [runtime, "run", "--rm", "-i", "--init", "--name", self.container_name]
        if runtime == "podman":
            cmd += ["--userns=keep-id"]
        else:
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        cmd += ["-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp/matplotlib",
                "-e", "NUMBA_CACHE_DIR=/tmp/numba", "-e", "PYTHONUNBUFFERED=1"]
        for name in c.get("env_passthrough", DEFAULT_ENV_PASSTHROUGH):
            if name in LLM_SECRET_VARS and not self.agent_code_llm_keys:
                continue
            if name in os.environ:
                cmd += ["-e", name]  # value read from our env; never on the command line
        if c.get("memory"):
            cmd += ["--memory", str(c["memory"]), "--memory-swap", str(c["memory"])]
        if c.get("cpus"):
            cmd += ["--cpus", str(c["cpus"])]
        cmd += ["--pids-limit", str(c.get("pids_limit", 4096))]
        if c.get("network", True) is False:
            cmd += ["--network", "none"]
        if c.get("gpus"):
            cmd += ["--gpus", str(c["gpus"])]
        # Image paths the agent must not read (e.g. build material and credentials baked into
        # the image): an empty read-only tmpfs is mounted over each.
        for p in c.get("mask_paths") or []:
            cmd += ["--tmpfs", f"{p}:ro,size=1k"]
        # Code and the output dir keep their absolute paths inside the container, so outputs,
        # plots and checkpoints mean the same thing on both sides. Inputs are mounted at
        # neutral paths under /inputs (see _input_mounts): their host paths name the cohort.
        cmd += ["-v", f"{self._code_mount()}:{self._code_mount()}:ro"]
        cmd += ["-v", f"{self.output_dir}:{self.output_dir}:rw", "-w", str(self.output_dir)]
        for host, ctr in self._input_mounts()[0]:
            cmd += ["-v", f"{host}:{ctr}:ro"]
        cmd += [c.get("image", DEFAULT_IMAGE), "python", worker_script]
        return cmd

    def _code_mount(self) -> Path:
        return self.repo_dir / "src" if self.mount == "code" else self.repo_dir

    # neutral in-container names per input key
    _INPUT_NAMES = {
        "scrna_data_path": "scrna", "scatac_data_path": "scatac",
        "scatac_snap_path": "scatac_backed/scatac_backed", "data_dir": "data_dir",
        "cnmf_usage_path": "cnmf_usage", "cnmf_spectra_scores_path": "cnmf_spectra_scores",
        "cnmf_top_genes_path": "cnmf_top_genes",
    }

    def _input_mounts(self) -> tuple[list[tuple[str, str]], dict]:
        """(host -> container mounts, data paths as the worker sees them).

        Only the input files themselves are mounted (resolved, so a symlink into the repo
        points at the real data), plus the per-sample files of a backed scATAC dataset at
        the same relative location. Nothing else of the data disk is visible."""
        data = self.init_payload.get("data", {}) or {}
        mounts, worker_data = [], {}
        for key, host in data.items():
            if not host:
                continue
            host = str(Path(host).resolve())
            name = self._INPUT_NAMES.get(key, key)
            ext = "" if Path(host).is_dir() else Path(host).suffix
            ctr = f"/inputs/{name}{ext}"
            mounts.append((host, ctr))
            worker_data[key] = ctr
            if key == "scatac_snap_path":
                host_base, ctr_base = Path(host).parent, Path(ctr).parent
                for comp in _backed_component_paths(host):
                    try:
                        rel = Path(comp).relative_to(host_base)
                    except ValueError:
                        raise WorkerFatal(f"backed scATAC component outside its dataset directory: {comp}")
                    mounts.append((comp, str(ctr_base / rel)))
        return mounts, worker_data

    def _worker_init_payload(self) -> dict:
        if self.mode != "container":
            return self.init_payload
        return dict(self.init_payload, data=self._input_mounts()[1])

    def _call_in_harness(self, name: str, args: dict, exec_id) -> dict:
        """Run a network tool (literature / database lookup) in the harness process."""
        flat = repr(args).lower()
        if any(h in flat for h in _BLOCKED_HOSTS):
            return {"success": False, "body": "[harness] Requests to the cloud metadata server are not allowed.",
                    "notes": [], "plots": [], "data_changes": "", "args_preview": args}
        if self._harness_runtime is None:
            from src.worker_runtime import ToolRuntime
            rt = ToolRuntime()
            rt.init(dict(self.init_payload, data={}, generation=0, checkpoints={},
                         secrets={k: os.environ[k] for k in LLM_SECRET_VARS if os.environ.get(k)}))
            self._harness_runtime = rt
            self._blacklist_terms = _blacklist_terms(self.init_payload.get("literature_blacklist"))
            log(f"🌐 Network tools run in the harness process: {', '.join(sorted(self.harness_tools))}")
        resp = self._harness_runtime.call({"name": name, "args": args, "exec_id": exec_id})
        # Database tools (e.g. GEO) do not apply the literature blacklist themselves; withhold any
        # result that mentions a blacklisted work, accession, URL or title.
        hit = _first_blacklist_hit(resp.get("body") or "", self._blacklist_terms)
        if hit:
            log(f"   🚫 {name} result withheld (matches blacklist entry {hit!r})")
            resp = dict(resp, body=("[harness] This result was withheld: it refers to sources that are "
                                    "excluded in this run. Use other sources."), plots=[])
        return resp

    def _spawn(self):
        cmd = self._command()
        env = os.environ.copy()
        if not self.agent_code_llm_keys:
            for name in LLM_SECRET_VARS:
                env.pop(name, None)
        if self.mode == "subprocess" and self.memory_limit_gb:
            env["M3A_WORKER_MEMORY_LIMIT_GB"] = str(self.memory_limit_gb)
        printable = " ".join(shlex.quote(x) for x in cmd)
        log(f"🧰 Starting tool worker (mode={self.mode}, generation={self.generation}): {printable}")
        self.proc = subprocess.Popen(
            cmd, cwd=str(self.output_dir), env=env,  # relative writes land in the run's output dir
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            start_new_session=True,  # Ctrl-C in the terminal must not hit the worker
        )
        self._in = self.proc.stdin.fileno()
        self._out = self.proc.stdout.fileno()
        hello = read_frame(self._out, timeout=float(self.container.get("startup_timeout_s", 600)))
        if not hello.get("ok"):
            raise WorkerFatal(f"worker failed to start: {hello}")

    def _request(self, req: dict, timeout: Optional[float]) -> dict:
        # If a previous request was interrupted mid-flight (Ctrl-C / SIGTERM in
        # the harness), the worker's next frame would be the stale answer:
        # never reuse such a worker.
        if self._busy:
            raise WorkerDied("worker is still busy with an interrupted request")
        self._busy = True
        write_frame(self._in, req)
        resp = read_frame(self._out, timeout=timeout)
        self._busy = False
        logs = resp.pop("logs", "")
        if logs:
            logger.write_file_only(logs)
        return resp

    def _kill(self):
        if self.proc is None:
            return
        if self.mode == "container" and self.container_name:
            subprocess.run([self.container.get("runtime", "docker"), "kill", self.container_name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        self.alive = False
        self._busy = False

    # ------------------------------------------------------------- lifecycle
    def start(self) -> dict:
        try:
            self._spawn()
            payload = dict(self._worker_init_payload(), op="init", generation=self.generation,
                           checkpoints=self._checkpoint_paths() if self.generation else {},
                           secrets={k: os.environ[k] for k in LLM_SECRET_VARS if os.environ.get(k)})
            resp = self._request(payload, timeout=None)
        except (WorkerDied, WorkerTimeout, OSError) as e:
            code = self.proc.poll() if self.proc else None
            self._kill()
            raise WorkerFatal(f"tool worker failed during startup: {e}; it {_describe_exit(code)}") from e
        if not resp.get("ok"):
            self._kill()
            raise WorkerFatal(f"tool worker could not load data: {resp.get('error')}\n{resp.get('traceback', '')}")
        self.alive = True
        self.info = resp.get("info", {})
        return self.info

    def _restart(self) -> str:
        self._kill()
        self.generation += 1
        ckpts = self._checkpoint_paths()
        self.start()
        if ckpts:
            return ("A new execution environment was started and the data was reloaded from the "
                    f"checkpoints taken after the last completed step ({', '.join(DATA_SLOTS[a] for a in ckpts)}"
                    "; anything else from the original inputs).")
        return "A new execution environment was started and the data was reloaded from the original inputs."

    def call(self, name: str, args: dict, exec_id) -> dict:
        if name in self.harness_tools:
            return self._call_in_harness(name, args, exec_id)
        timeout = (float(self.tool_timeout_s) + self.grace) if self.tool_timeout_s else None
        try:
            resp = self._request({"op": "call", "name": name, "args": args, "exec_id": exec_id}, timeout)
        except (WorkerDied, WorkerTimeout) as e:
            if isinstance(e, WorkerTimeout):
                cause = (f"did not finish within the hard limit of {timeout:.0f}s "
                         f"(it was stuck in code that could not be interrupted) and was killed")
                code = None
            else:
                try:
                    code = self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    code = None
                cause = _describe_exit(code)
            log(f"   💥 Tool worker {cause} during {name}; restarting...")
            restart_note = self._restart()  # raises WorkerFatal if it cannot come back
            return {
                "success": False,
                "body": (f"[harness] The execution environment {cause} while running this call. "
                         f"The call's effects are lost. {restart_note} Variables you defined in "
                         f"run_python_repl no longer exist and must be recreated. If this was caused "
                         f"by memory use, process the data in smaller pieces or avoid densifying "
                         f"large matrices (e.g. .X.toarray())."),
                "notes": [], "plots": [], "data_changes": "", "args_preview": args,
                "worker_restarted": True,
            }
        if not resp.get("ok"):
            # Worker-side exception outside the tool itself (harness bug): report, keep going
            return {"success": False,
                    "body": f"[harness] internal error while running the tool: {resp.get('error')}",
                    "notes": [], "plots": [], "data_changes": "", "args_preview": args}
        return resp

    def checkpoint(self) -> dict:
        if not self.alive:
            return {}
        try:
            resp = self._request({"op": "checkpoint"}, timeout=None)
        except (WorkerDied, WorkerTimeout) as e:
            log(f"⚠️ Tool worker died while checkpointing ({e}); restarting from previous checkpoints")
            self._restart()
            return {}
        return resp.get("results", {}) if resp.get("ok") else {"_error": resp.get("error")}

    def finalize(self) -> dict:
        """Ask the worker for final outputs. If it is gone, fall back to the
        latest checkpoints (hard-linked/copied to result_*.h5ad)."""
        if self.alive and not self._busy and self.proc is not None and self.proc.poll() is None:
            try:
                resp = self._request({"op": "finalize"}, timeout=3600)
                if resp.get("ok"):
                    return resp
            except (WorkerDied, WorkerTimeout) as e:
                log(f"⚠️ Tool worker unavailable for final save ({e})")
        return self._finalize_from_checkpoints()

    def _finalize_from_checkpoints(self) -> dict:
        import shutil
        saved, errors = {}, {}
        names = {"adata_rna": "result_rna.h5ad", "adata_atac": "result_atac.h5ad",
                 "adata_combined": "result_combined.h5ad"}
        for attr, path in self._checkpoint_paths().items():
            if attr not in names:
                continue
            dst = self.output_dir / names[attr]
            try:
                shutil.copy2(path, dst)
                saved[attr] = str(dst) + " (from last checkpoint)"
            except Exception as e:
                errors[attr] = str(e)
        return {"saved": saved, "errors": errors, "info": self.info}

    def shutdown(self):
        if self.proc is None:
            return
        if self.proc.poll() is None and not self._busy:
            try:
                self._request({"op": "shutdown"}, timeout=30)
                self.proc.wait(timeout=30)
            except Exception:
                pass
        self._kill()


def make_runtime_client(config: dict, init_payload: dict, output_dir: Path, repo_dir: Path,
                        tool_timeout_s=None):
    ex = config.get("execution", {}) or {}
    mode = ex.get("mode", "subprocess")
    if mode == "inprocess":
        return InProcessClient(init_payload, output_dir, tool_timeout_s)
    if mode not in ("subprocess", "container"):
        raise ValueError(f"execution.mode must be subprocess | container | inprocess, got {mode!r}")
    container = ex.get("container", {}) or {}
    harness_tools = frozenset()
    # No network in the worker: literature/database tools (fixed code, no data access) run in
    # the harness instead, so agent-written code has no route to the internet.
    if mode == "container" and container.get("network", True) is False \
            and ex.get("network_tools_in_harness", True):
        harness_tools = network_tool_names()
    return WorkerClient(
        init_payload, output_dir, repo_dir, mode=mode, tool_timeout_s=tool_timeout_s,
        hard_timeout_grace_s=ex.get("hard_timeout_grace_s", 120),
        container=container,
        memory_limit_gb=ex.get("memory_limit_gb"),
        agent_code_llm_keys=ex.get("agent_code_llm_keys", False),
        harness_tools=harness_tools,
    )


NETWORK_TOOL_CATEGORIES = ("external_literature", "external_database")


def network_tool_names() -> frozenset:
    from src.tool_registry import get_tools
    return frozenset(t["name"] for cat in NETWORK_TOOL_CATEGORIES for t in get_tools(cat))
