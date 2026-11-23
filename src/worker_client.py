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


class WorkerClient(_BaseClient):
    def __init__(self, init_payload: dict, output_dir: Path, repo_dir: Path, mode: str = "subprocess",
                 tool_timeout_s=None, hard_timeout_grace_s: float = 120.0, container: dict = None,
                 memory_limit_gb=None):
        super().__init__(init_payload, output_dir, tool_timeout_s)
        self.repo_dir = Path(repo_dir).resolve()
        self.mode = mode
        self.grace = float(hard_timeout_grace_s)
        self.container = container or {}
        self.memory_limit_gb = memory_limit_gb
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
        # Same absolute paths inside and outside the container, so every path
        # (outputs, plots, checkpoints) means the same thing on both sides.
        cmd += ["-v", f"{self.repo_dir}:{self.repo_dir}:ro"]
        cmd += ["-v", f"{self.output_dir}:{self.output_dir}:rw", "-w", str(self.output_dir)]
        for p in self._input_paths():
            cmd += ["-v", f"{p}:{p}:ro"]
        cmd += [c.get("image", DEFAULT_IMAGE), "python", worker_script]
        return cmd

    def _input_paths(self) -> list[str]:
        data = self.init_payload.get("data", {}) or {}
        paths = sorted({str(Path(v).resolve()) for v in data.values() if v})
        out_s = str(self.output_dir)
        # Skip paths already covered by another mount (or by the output dir)
        kept = []
        for p in paths:
            if p.startswith(out_s + os.sep) or p.startswith(str(self.repo_dir) + os.sep):
                continue
            if any(p.startswith(k + os.sep) for k in kept):
                continue
            kept.append(p)
        return kept

    def _spawn(self):
        cmd = self._command()
        env = os.environ.copy()
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
            payload = dict(self.init_payload, op="init", generation=self.generation,
                           checkpoints=self._checkpoint_paths() if self.generation else {})
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
    return WorkerClient(
        init_payload, output_dir, repo_dir, mode=mode, tool_timeout_s=tool_timeout_s,
        hard_timeout_grace_s=ex.get("hard_timeout_grace_s", 120),
        container=ex.get("container", {}),
        memory_limit_gb=ex.get("memory_limit_gb"),
    )
