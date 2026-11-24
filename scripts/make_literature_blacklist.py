#!/usr/bin/env python
"""Build a literature/code blacklist for the HTAN multiome evaluation tasks.

The harness's literature tools (src/tools_external/literature.py) drop any result
that matches ``literature_blacklist`` (DOIs / URLs substring match, PMIDs exact,
PMCIDs, title_contains substring).  This script generates that blacklist so an
agent cannot shortcut the evaluation by reading the source study, its
supplementary tables, papers that cite/re-use the data, or the code repos that
hold the original annotations.

Pipeline
--------
1. Source identification: Claude proposes candidate source/companion papers for
   the dataset description; every candidate is verified against OpenAlex (DOI
   lookup or title search + title similarity).  The source paper's full text
   (PMC via NCBI E-utilities) is checked for dataset fingerprints (HTAN, cancer
   types).  Preprints are searched via the bioRxiv "pubs" API, Crossref
   posted-content and OpenAlex/Semantic Scholar title search.
2. Citing papers: OpenAlex ``cites:`` (cursor pagination) + Semantic Scholar
   citations + NCBI PubMed "cited in", merged by DOI/PMID; PMCIDs filled via the
   PMC ID converter.  Optionally Claude flags which citing papers re-use the data.
3. Code: GitHub URLs from the paper's full text, the authoring org's repos
   (keyword-filtered) and GitHub repo search; Claude decides which are
   study/annotation code vs generic tools; each is verified via the GitHub REST
   API and all forks are enumerated recursively.
4. Data portals: accessions in the paper's Data-availability section, PubMed ->
   GEO links, CELLxGENE collections and Zenodo records linked to the DOI.

Outputs: ``config/htan/literature_blacklist.yaml`` (keys pmids, pmcids, dois,
urls, title_contains) and ``config/htan/literature_blacklist.provenance.json``.

Usage
-----
    .venv/bin/python scripts/make_literature_blacklist.py                  # full run
    .venv/bin/python scripts/make_literature_blacklist.py --max-citing 20  # quick test
    .venv/bin/python scripts/make_literature_blacklist.py --flag-reuse     # + LLM reuse flags
    .venv/bin/python scripts/make_literature_blacklist.py --no-llm --source-doi 10.1038/s41586-023-06682-5

Environment: ANTHROPIC_API_KEY (Claude), optional GITHUB_TOKEN (else `gh auth
token`, else unauthenticated), optional OPENALEX_API_KEY, S2_API_KEY.
No personal contact info is sent to any API (generic User-Agent, no mailto).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "htan-blacklist-builder/1.0"
TOOL_NAME = "htan-blacklist-builder"

DEFAULT_DATASET_DESCRIPTION = (
    "HTAN WUSTL atlas (Human Tumor Atlas Network, Washington University in St. Louis, Ding lab): "
    "single-cell / single-nucleus multiome data (scRNA-seq/snRNA-seq + snATAC-seq) from 11 cancer types "
    "(PDAC, BRCA, HNSCC, CRC, SKCM, CESC, MM, UCEC, OV, GBM, ccRCC), 145 patients / 161 biospecimens. "
    "HTAN sample IDs look like HT029B1-S1PC, CE357E1-S1, VF031V1-Tm1Y1, ML1254M1-Ty1, CPT704DU-S1. "
    "Tasks ask to annotate cancer type, cancer site, cell types, and rank cNMF programs by relevance to "
    "molecular/clinical endpoints."
)
DEFAULT_OUT = REPO_ROOT / "config" / "htan" / "literature_blacklist.yaml"
DEFAULT_PROV = REPO_ROOT / "config" / "htan" / "literature_blacklist.provenance.json"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "htan_blacklist_cache"

# Keywords used to pre-filter an authoring org's GitHub repos before Claude adjudicates them.
REPO_KEYWORDS = r"atac|pancan|htan|multiome|snrna|scrna|single.?cell|cnmf|epigen|publication|atlas"

PREPRINT_HOST_RE = re.compile(r"biorxiv|medrxiv|researchsquare|ssrn|arxiv|preprints\.org|authorea|qeios", re.I)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def clean_title(t: Optional[str]) -> str:
    if not t:
        return ""
    t = html.unescape(re.sub(r"<[^>]+>", "", str(t)))
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.rstrip(".").strip()


def title_key(t: Optional[str]) -> str:
    t = clean_title(t).lower()
    t = t.replace("tumour", "tumor").replace("tumours", "tumors")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def title_sim(a: Optional[str], b: Optional[str]) -> float:
    ka, kb = title_key(a), title_key(b)
    if not ka or not kb:
        return 0.0
    return difflib.SequenceMatcher(None, ka, kb).ratio()


def norm_doi(d: Optional[str]) -> str:
    if not d:
        return ""
    d = str(d).strip().lower()
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    return d.strip().rstrip(".")


def norm_pmid(p: Any) -> str:
    if not p:
        return ""
    m = re.search(r"(\d+)\s*/?$", str(p).strip())
    return m.group(1) if m else ""


def norm_pmcid(p: Any) -> str:
    if not p:
        return ""
    m = re.search(r"(?:pmc)?(\d+)\s*/?$", str(p).strip(), re.I)
    return f"PMC{m.group(1)}" if m else ""


def norm_url(u: Optional[str]) -> str:
    """Lowercase, scheme/www/query/fragment/trailing-slash stripped URL for substring matching."""
    if not u:
        return ""
    p = urlparse(u.strip())
    if not p.netloc:
        return ""
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/")
    return f"{host}{path}".lower()


def url_is_specific(u: str) -> bool:
    """Reject bare domains / very short paths that would block unrelated content."""
    if "/" not in u:
        return False
    host, path = u.split("/", 1)
    if host in {"doi.org", "dx.doi.org"}:  # DOI entries already cover these
        return False
    return len(path) >= 6 and bool(re.search(r"\d", path) or len(path) >= 15)


def doi_suffix_pattern(doi: str) -> Optional[str]:
    """Distinctive DOI suffix (e.g. s41586-023-06682-5) to catch publisher URLs w/o the 10.x prefix."""
    if "/" not in doi:
        return None
    suf = doi.split("/", 1)[1]
    if len(suf) >= 12 and re.search(r"\d{3,}", suf) and re.search(r"[a-z.\-]", suf):
        return suf
    return None


# ---------------------------------------------------------------------------
# Cached, rate-limited HTTP
# ---------------------------------------------------------------------------

class Http:
    MIN_INTERVAL = {
        "api.openalex.org": 0.15,
        "api.semanticscholar.org": 1.1,
        "eutils.ncbi.nlm.nih.gov": 0.4,
        "pmc.ncbi.nlm.nih.gov": 0.4,
        "api.github.com": 0.3,
        "api.crossref.org": 0.3,
        "api.biorxiv.org": 0.3,
        "zenodo.org": 1.0,
    }

    def __init__(self, cache_dir: Path, refresh: bool = False, max_wait: float = 900.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.max_wait = max_wait
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last: dict[str, float] = {}
        self.stats = defaultdict(int)

    def _key(self, method: str, url: str, params: Optional[dict], body: Any) -> Path:
        blob = json.dumps([method, url, sorted((params or {}).items()), body], sort_keys=True, default=str)
        h = hashlib.sha256(blob.encode()).hexdigest()
        host = urlparse(url).netloc.replace(":", "_") or "misc"
        d = self.cache_dir / host
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{h}.json"

    def _throttle(self, host: str) -> None:
        iv = self.MIN_INTERVAL.get(host, 0.2)
        dt = time.time() - self._last.get(host, 0.0)
        if dt < iv:
            time.sleep(iv - dt)
        self._last[host] = time.time()

    def request(
        self,
        url: str,
        *,
        params: Optional[dict] = None,
        method: str = "GET",
        json_body: Any = None,
        headers: Optional[dict] = None,
        secret_params: Optional[dict] = None,
        as_json: bool = True,
        retries: int = 6,
    ) -> Optional[Any]:
        """Return parsed JSON (or text). Returns None on 404/410/400/422. Caches successes and not-founds.
        ``secret_params``/``headers`` are sent but excluded from the cache key."""
        path = self._key(method, url, params, json_body)
        if path.exists() and not self.refresh:
            self.stats["cache_hit"] += 1
            rec = json.loads(path.read_text())
            return rec.get("data")
        host = urlparse(url).netloc
        all_params = dict(params or {})
        all_params.update(secret_params or {})
        delay = 2.0
        for attempt in range(retries):
            self._throttle(host)
            try:
                r = self.session.request(method, url, params=all_params or None, json=json_body,
                                         headers=headers, timeout=60)
            except requests.RequestException as e:
                log(f"  [http] {host}: {e.__class__.__name__}; retry in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            self.stats[f"net:{host}"] += 1
            if r.status_code in (400, 404, 410, 422):
                path.write_text(json.dumps({"url": url, "status": r.status_code, "data": None}))
                return None
            gh_limited = host == "api.github.com" and r.status_code in (403, 429) and \
                r.headers.get("X-RateLimit-Remaining") == "0"
            if r.status_code == 429 or gh_limited or r.status_code >= 500 or \
                    (host == "api.github.com" and r.status_code == 403 and "rate limit" in r.text.lower()):
                wait = delay
                if r.headers.get("Retry-After", "").isdigit():
                    wait = float(r.headers["Retry-After"])
                elif r.headers.get("X-RateLimit-Reset", "").isdigit() and gh_limited:
                    wait = max(1.0, float(r.headers["X-RateLimit-Reset"]) - time.time() + 2)
                if wait > self.max_wait:
                    raise RuntimeError(f"Rate limited by {host}; would need to wait {wait:.0f}s "
                                       f"(> --max-wait {self.max_wait:.0f}s). Set a token or rerun later.")
                log(f"  [http] {host} HTTP {r.status_code}; backing off {wait:.0f}s")
                time.sleep(wait)
                delay = min(delay * 2, 120)
                continue
            if r.status_code >= 300:
                log(f"  [http] {host} HTTP {r.status_code} for {url}; giving up")
                return None
            data = r.json() if as_json else r.text
            path.write_text(json.dumps({"url": url, "params": params, "status": r.status_code, "data": data}))
            return data
        log(f"  [http] {host}: retries exhausted for {url}")
        return None


# ---------------------------------------------------------------------------
# Scholarly APIs
# ---------------------------------------------------------------------------

class Scholarly:
    OA = "https://api.openalex.org"
    S2 = "https://api.semanticscholar.org/graph/v1"
    EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    OA_FIELDS = ("id,doi,title,display_name,ids,locations,primary_location,publication_year,type,"
                 "abstract_inverted_index,referenced_works")

    def __init__(self, http: Http, openalex_key: str = "", mailto: str = "", s2_key: str = ""):
        self.http = http
        self.oa_secret = {"api_key": openalex_key} if openalex_key else {}
        if mailto:
            self.oa_secret["mailto"] = mailto
        self.s2_headers = {"x-api-key": s2_key} if s2_key else None

    # ---- OpenAlex
    def oa_get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        return self.http.request(f"{self.OA}{path}", params=params, secret_params=self.oa_secret)

    def oa_by_doi(self, doi: str) -> Optional[dict]:
        return self.oa_get(f"/works/doi:{doi}", {"select": self.OA_FIELDS})

    def oa_title_search(self, title: str, n: int = 10) -> list[dict]:
        q = re.sub(r"[,:|!()\"']", " ", clean_title(title))
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            return []
        d = self.oa_get("/works", {"filter": f"title.search:{q}", "per-page": n, "select": self.OA_FIELDS})
        return (d or {}).get("results", [])

    def oa_search(self, query: str, n: int = 10) -> list[dict]:
        d = self.oa_get("/works", {"search": query, "per-page": n, "select": self.OA_FIELDS})
        return (d or {}).get("results", [])

    def oa_citing(self, work_id: str, limit: Optional[int]) -> list[dict]:
        """All works citing ``work_id``. Uses cursor paging with an API key; without one OpenAlex only
        allows page= paging (first 10,000 results), so that is used instead."""
        wid = work_id.rsplit("/", 1)[-1]
        use_cursor = "api_key" in self.oa_secret
        out, cursor, page = [], "*", 1
        while True:
            params = {"filter": f"cites:{wid}", "per-page": 200, "select": self.OA_FIELDS,
                      "sort": "publication_date:asc"}
            params.update({"cursor": cursor} if use_cursor else {"page": page})
            d = self.oa_get("/works", params)
            if not d:
                raise RuntimeError(f"OpenAlex citing query failed for {wid} (page {page}); rerun later")
            res = d.get("results", [])
            out.extend(res)
            total = (d.get("meta") or {}).get("count", 0)
            cursor = (d.get("meta") or {}).get("next_cursor")
            page += 1
            if not res or len(out) >= total or (limit and len(out) >= limit) or (use_cursor and not cursor) \
                    or (not use_cursor and len(out) >= 10000):
                break
        return out

    # ---- Semantic Scholar
    S2_FIELDS = "title,externalIds,url,year,abstract"

    def s2_paper(self, doi: str) -> Optional[dict]:
        return self.http.request(f"{self.S2}/paper/DOI:{doi}", params={"fields": self.S2_FIELDS},
                                 headers=self.s2_headers)

    def s2_match(self, title: str) -> Optional[dict]:
        d = self.http.request(f"{self.S2}/paper/search/match",
                              params={"query": clean_title(title), "fields": self.S2_FIELDS},
                              headers=self.s2_headers)
        return ((d or {}).get("data") or [None])[0]

    def s2_citing(self, doi: str, limit: Optional[int]) -> list[dict]:
        out, offset = [], 0
        while True:
            page = 1000 if not limit else min(1000, max(1, limit - len(out)))
            d = self.http.request(f"{self.S2}/paper/DOI:{doi}/citations",
                                  params={"fields": self.S2_FIELDS, "limit": page, "offset": offset},
                                  headers=self.s2_headers)
            if not d:
                break
            out.extend(x["citingPaper"] for x in d.get("data", []) if x.get("citingPaper"))
            if d.get("next") is None or (limit and len(out) >= limit):
                break
            offset = d["next"]
        return out

    # ---- NCBI
    def ncbi_citedin(self, pmid: str) -> list[str]:
        d = self.http.request(f"{self.EU}/elink.fcgi", params={
            "dbfrom": "pubmed", "linkname": "pubmed_pubmed_citedin", "id": pmid, "retmode": "json",
            "tool": TOOL_NAME})
        ids: list[str] = []
        for ls in (d or {}).get("linksets", []):
            for db in ls.get("linksetdbs", []):
                ids.extend(db.get("links", []))
        return sorted(set(ids), key=int)

    def ncbi_pubmed_summaries(self, pmids: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i in range(0, len(pmids), 200):
            chunk = pmids[i:i + 200]
            d = self.http.request(f"{self.EU}/esummary.fcgi", method="POST", params={
                "db": "pubmed", "id": ",".join(chunk), "retmode": "json", "tool": TOOL_NAME})
            res = (d or {}).get("result", {})
            for uid in res.get("uids", []):
                out[uid] = res[uid]
        return out

    def ncbi_gds_links(self, pmid: str) -> list[str]:
        d = self.http.request(f"{self.EU}/elink.fcgi", params={
            "dbfrom": "pubmed", "db": "gds", "id": pmid, "retmode": "json", "tool": TOOL_NAME})
        uids = []
        for ls in (d or {}).get("linksets", []):
            for db in ls.get("linksetdbs", []):
                uids.extend(db.get("links", []))
        if not uids:
            return []
        s = self.http.request(f"{self.EU}/esummary.fcgi", params={
            "db": "gds", "id": ",".join(sorted(uids)), "retmode": "json", "tool": TOOL_NAME})
        res = (s or {}).get("result", {})
        return sorted({res[u].get("accession") for u in res.get("uids", []) if res[u].get("accession")})

    def ncbi_gds_exists(self, acc: str) -> bool:
        d = self.http.request(f"{self.EU}/esearch.fcgi", params={
            "db": "gds", "term": f"{acc}[ACCN]", "retmode": "json", "tool": TOOL_NAME})
        return int(((d or {}).get("esearchresult") or {}).get("count", 0)) > 0

    def pmc_fulltext(self, pmcid: str) -> str:
        num = pmcid.upper().replace("PMC", "")
        t = self.http.request(f"{self.EU}/efetch.fcgi", params={"db": "pmc", "id": num, "tool": TOOL_NAME},
                              as_json=False)
        return t or ""

    def ncbi_ids_for_dois(self, dois: list[str]) -> dict[str, dict]:
        """DOI -> {pmid, pmcid, doi} via PubMed esearch ([AID]) + esummary.  (The PMC ID converter blocks
        python-requests clients with HTTP 403, so E-utilities are used instead.)"""
        out: dict[str, dict] = {}
        for i in range(0, len(dois), 40):
            chunk = [d for d in dois[i:i + 40] if '"' not in d]
            term = " OR ".join(f'"{d}"[AID]' for d in chunk)
            d = self.http.request(f"{self.EU}/esearch.fcgi", params={
                "db": "pubmed", "term": term, "retmax": 500, "retmode": "json", "tool": TOOL_NAME})
            pmids = ((d or {}).get("esearchresult") or {}).get("idlist", [])
            want = set(chunk)
            for uid, s in self.ncbi_pubmed_summaries(sorted(pmids, key=int)).items():
                rec = paper_from_pubmed(uid, s)
                if rec["doi"] in want:
                    out[rec["doi"]] = {"pmid": uid, "pmcid": rec["pmcid"], "doi": rec["doi"]}
        return out

    def ncbi_ids_for_pmids(self, pmids: list[str]) -> dict[str, dict]:
        return {uid: {"pmid": uid, "pmcid": r["pmcid"], "doi": r["doi"]}
                for uid, s in self.ncbi_pubmed_summaries(pmids).items() for r in [paper_from_pubmed(uid, s)]}

    # ---- Preprint servers
    def biorxiv_pubs(self, doi: str) -> list[dict]:
        out = []
        for server in ("biorxiv", "medrxiv"):
            d = self.http.request(f"https://api.biorxiv.org/pubs/{server}/{doi}")
            for c in (d or {}).get("collection", []):
                c["_server"] = server
                out.append(c)
        return out

    def crossref_posted(self, title: str) -> list[dict]:
        d = self.http.request("https://api.crossref.org/works", params={
            "query.title": clean_title(title), "filter": "type:posted-content", "rows": 10,
            "select": "DOI,title,type,relation,URL"})
        return ((d or {}).get("message") or {}).get("items", [])


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

class Claude:
    def __init__(self, model: str, http_cache: Path, enabled: bool = True, effort: str = "high"):
        self.model = model
        self.enabled = enabled
        self.effort = effort
        self.cache_dir = http_cache / "anthropic"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.disabled_reason = "" if enabled else "--no-llm"
        self._client = None
        if enabled:
            import anthropic  # noqa: WPS433
            self._anthropic = anthropic
            self._client = anthropic.Anthropic(max_retries=4)

    def ask_json(self, purpose: str, system: str, prompt: str, schema: dict, max_tokens: int = 16000) -> Optional[dict]:
        if not self.enabled:
            return None
        key = hashlib.sha256(json.dumps([self.model, self.effort, system, prompt, schema], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            self.calls.append({"purpose": purpose, "cached": True})
            return data
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        a = self._anthropic
        resp = None
        try:
            # Server-side refusal fallback (beta) - falls back to plain request if not accepted.
            resp = self._client.messages.create(
                **kwargs, extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
                extra_body={"fallbacks": "default"})
        except a.BadRequestError as e:
            log(f"  [claude] fallback-enabled request rejected ({e.message[:120]}); retrying without fallbacks")
            resp = self._client.messages.create(**kwargs)
        except (a.AuthenticationError, a.PermissionDeniedError) as e:
            self.enabled = False
            self.disabled_reason = f"{e.__class__.__name__} (HTTP {getattr(e, 'status_code', '?')})"
            log(f"  [claude] {purpose}: {self.disabled_reason} -> LLM disabled for this run")
            self.calls.append({"purpose": purpose, "error": self.disabled_reason})
            return None
        except (a.RateLimitError, a.APIConnectionError, a.APIStatusError) as e:
            log(f"  [claude] {purpose}: API error {e.__class__.__name__}: {str(e)[:200]}")
            return None
        if resp.stop_reason == "refusal":
            log(f"  [claude] {purpose}: refusal ({getattr(resp, 'stop_details', None)})")
            return None
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log(f"  [claude] {purpose}: could not parse JSON (stop_reason={resp.stop_reason})")
            return None
        u = resp.usage
        self.calls.append({"purpose": purpose, "cached": False, "model": resp.model,
                           "input_tokens": u.input_tokens, "output_tokens": u.output_tokens})
        path.write_text(json.dumps(data))
        return data


CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "papers": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "first_author": {"type": "string"},
                "year": {"type": "integer"},
                "venue": {"type": "string"},
                "doi": {"type": "string"},
                "role": {"type": "string", "enum": ["source", "preprint", "companion", "data_descriptor"]},
                "rationale": {"type": "string"},
            },
            "required": ["title", "first_author", "year", "venue", "doi", "role", "rationale"],
            "additionalProperties": False,
        }},
        "code_repos": {"type": "array", "items": {"type": "string"}},
        "data_resources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["papers", "code_repos", "data_resources"],
    "additionalProperties": False,
}

REPO_SCHEMA = {
    "type": "object",
    "properties": {"repos": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "decision": {"type": "string", "enum": ["study_code", "related_study_code", "generic_tool", "unrelated"]},
            "reason": {"type": "string"},
        },
        "required": ["repo", "decision", "reason"],
        "additionalProperties": False,
    }}},
    "required": ["repos"],
    "additionalProperties": False,
}

REUSE_SCHEMA = {
    "type": "object",
    "properties": {"papers": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "reuse": {"type": "string", "enum": ["yes", "likely", "no", "unclear"]},
            "reason": {"type": "string"},
        },
        "required": ["id", "reuse", "reason"],
        "additionalProperties": False,
    }}},
    "required": ["papers"],
    "additionalProperties": False,
}

SYSTEM = ("You are a careful biomedical bibliographer helping build an exclusion list so that an AI agent being "
          "evaluated on a public dataset cannot read the dataset's original publication or derived annotations. "
          "Only report items you are confident exist; leave DOI empty rather than guess. Answer in the JSON schema.")


# ---------------------------------------------------------------------------
# Paper records
# ---------------------------------------------------------------------------

def oa_abstract(w: dict) -> str:
    inv = w.get("abstract_inverted_index") or {}
    if not inv:
        return ""
    pos = sorted((p, word) for word, ps in inv.items() for p in ps)
    return " ".join(word for _, word in pos)


def paper_from_openalex(w: dict) -> dict:
    ids = w.get("ids") or {}
    urls, dois = set(), set()
    for loc in (w.get("locations") or []) + [w.get("primary_location") or {}]:
        if not loc:
            continue
        for k in ("landing_page_url",):
            if loc.get(k):
                urls.add(loc[k])
        if str(loc.get("id", "")).startswith("doi:"):
            dois.add(norm_doi(loc["id"][4:]))
        lp = loc.get("landing_page_url") or ""
        if "doi.org/" in lp:
            dois.add(norm_doi(lp))
    if w.get("doi"):
        dois.add(norm_doi(w["doi"]))
    if ids.get("pmcid"):
        urls.add(ids["pmcid"])
    return {
        "openalex": w.get("id"),
        "doi": norm_doi(w.get("doi")),
        "extra_dois": sorted(d for d in dois if d and d != norm_doi(w.get("doi"))),
        "pmid": norm_pmid(ids.get("pmid")),
        "pmcid": norm_pmcid(ids.get("pmcid")),
        "title": clean_title(w.get("title") or w.get("display_name")),
        "year": w.get("publication_year"),
        "type": w.get("type"),
        "urls": sorted(urls),
        "abstract": oa_abstract(w),
        "apis": {"openalex"},
    }


def paper_from_s2(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    return {
        "s2": p.get("paperId"),
        "doi": norm_doi(ext.get("DOI")),
        "extra_dois": [],
        "pmid": norm_pmid(ext.get("PubMed")),
        "pmcid": norm_pmcid(ext.get("PubMedCentral")),
        "title": clean_title(p.get("title")),
        "year": p.get("year"),
        "urls": [],
        "abstract": p.get("abstract") or "",
        "apis": {"semantic_scholar"},
        "arxiv": ext.get("ArXiv"),
    }


def paper_from_pubmed(uid: str, s: dict) -> dict:
    aid = {a.get("idtype"): a.get("value") for a in s.get("articleids", [])}
    return {
        "doi": norm_doi(aid.get("doi")),
        "extra_dois": [],
        "pmid": uid,
        "pmcid": norm_pmcid(aid.get("pmc")),
        "title": clean_title(s.get("title")),
        "year": (s.get("pubdate") or "")[:4],
        "urls": [],
        "abstract": "",
        "apis": {"ncbi_pubmed"},
    }


def merge_papers(papers: Iterable[dict]) -> list[dict]:
    """Union-find style merge on DOI / PMID / PMCID / normalised title."""
    merged: list[dict] = []
    index: dict[str, int] = {}

    def keys(p: dict) -> list[str]:
        k = []
        for d in [p.get("doi")] + list(p.get("extra_dois") or []):
            if d:
                k.append("doi:" + d)
        if p.get("pmid"):
            k.append("pmid:" + p["pmid"])
        if p.get("pmcid"):
            k.append("pmcid:" + p["pmcid"])
        tk = title_key(p.get("title"))
        if len(tk) >= 40:
            k.append("t:" + tk)
        return k

    for p in papers:
        ks = keys(p)
        hits = sorted({index[k] for k in ks if k in index})
        if not hits:
            merged.append({**p, "apis": set(p.get("apis") or []), "extra_dois": list(p.get("extra_dois") or []),
                           "urls": list(p.get("urls") or [])})
            idx = len(merged) - 1
        else:
            idx = hits[0]
            tgt = merged[idx]
            for other in hits[1:]:  # fold additional matched records into the first
                _fold(tgt, merged[other])
                merged[other] = None  # type: ignore
                for kk, vv in list(index.items()):
                    if vv == other:
                        index[kk] = idx
            _fold(tgt, p)
        for k in keys(merged[idx]):
            index[k] = idx
    return [m for m in merged if m]


def _fold(tgt: dict, src: dict) -> None:
    for f in ("doi", "pmid", "pmcid", "title", "year", "openalex", "s2", "abstract", "type"):
        if not tgt.get(f) and src.get(f):
            tgt[f] = src[f]
    if src.get("doi") and tgt.get("doi") and src["doi"] != tgt["doi"]:
        tgt["extra_dois"] = list(tgt.get("extra_dois") or []) + [src["doi"]]
    tgt["extra_dois"] = sorted(set(tgt.get("extra_dois") or []) | set(src.get("extra_dois") or []) - {tgt.get("doi")})
    tgt["urls"] = sorted(set(tgt.get("urls") or []) | set(src.get("urls") or []))
    tgt["apis"] = set(tgt.get("apis") or []) | set(src.get("apis") or [])


# ---------------------------------------------------------------------------
# Blacklist accumulator with provenance
# ---------------------------------------------------------------------------

class Blacklist:
    def __init__(self, min_title_len: int):
        self.min_title_len = min_title_len
        self.entries: dict[str, dict[str, list[dict]]] = {k: {} for k in ("pmids", "pmcids", "dois", "urls", "title_contains")}
        self.skipped_titles: list[dict] = []

    def add(self, key: str, value: str, category: str, why: str, apis: Iterable[str]) -> None:
        if not value:
            return
        if key == "title_contains":
            value = clean_title(value)
            if len(value) < self.min_title_len or len(value.split()) < 4:
                self.skipped_titles.append({"title": value, "category": category, "why": why})
                return
        if key == "dois" and (len(value) < 10 or "/" not in value):
            return
        rec = {"category": category, "why": why, "apis": sorted(set(apis))}
        lst = self.entries[key].setdefault(value, [])
        if rec not in lst:
            lst.append(rec)

    def add_paper(self, p: dict, category: str, why: str, strong: bool = False) -> None:
        apis = p.get("apis") or []
        for d in [p.get("doi")] + list(p.get("extra_dois") or []):
            if d:
                self.add("dois", d, category, why, apis)
                if strong:
                    suf = doi_suffix_pattern(d)
                    if suf:
                        self.add("urls", suf, category, f"{why} (DOI suffix in publisher URLs)", apis)
        if p.get("pmid"):
            self.add("pmids", p["pmid"], category, why, apis)
            self.add("urls", f"pubmed.ncbi.nlm.nih.gov/{p['pmid']}", category, why, apis)
            if strong:
                for pat in ("europepmc.org/article/med/{}", "europepmc.org/abstract/med/{}",
                            "ncbi.nlm.nih.gov/pubmed/{}"):
                    self.add("urls", pat.format(p["pmid"]), category, why, apis)
        if p.get("pmcid"):
            self.add("pmcids", p["pmcid"], category, why, apis)
            num = p["pmcid"][3:]
            # OpenAlex-style PMC URLs carry the bare number (no "PMC" prefix).
            self.add("urls", f"ncbi.nlm.nih.gov/pmc/articles/{num}", category, why, apis)
        for u in p.get("urls") or []:
            nu = norm_url(u)
            if nu and url_is_specific(nu):
                self.add("urls", nu, category, why, apis)
        if p.get("title"):
            self.add("title_contains", p["title"], category, why, apis)

    def to_yaml_dict(self) -> dict[str, list[str]]:
        def sort(vals: Iterable[str]) -> list[str]:
            return sorted(set(vals), key=lambda s: (s.lower(), s))
        return {
            "pmids": sorted(self.entries["pmids"], key=int),
            "pmcids": sort(self.entries["pmcids"]),
            "dois": sort(self.entries["dois"]),
            "urls": sort(self.entries["urls"]),
            "title_contains": sort(self.entries["title_contains"]),
        }


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def verify_candidate(sch: Scholarly, cand: dict) -> tuple[Optional[dict], str]:
    """Return (openalex work, evidence) if the candidate can be verified in OpenAlex."""
    doi = norm_doi(cand.get("doi"))
    if doi:
        w = sch.oa_by_doi(doi)
        if w:
            s = title_sim(w.get("title"), cand.get("title"))
            if s >= 0.75:
                return w, f"OpenAlex DOI lookup; title similarity {s:.2f}"
    best, best_s = None, 0.0
    for w in sch.oa_title_search(cand.get("title", "")):
        s = title_sim(w.get("title"), cand.get("title"))
        if s > best_s:
            best, best_s = w, s
    if best and best_s >= 0.9:
        return best, f"OpenAlex title search; title similarity {best_s:.2f}"
    return None, f"not verified (best OpenAlex title similarity {best_s:.2f})"


def fingerprint_fulltext(text: str) -> dict:
    t = text
    types = {"PDAC": r"pancrea", "BRCA": r"breast", "HNSCC": r"head and neck", "CRC": r"colorectal|colon",
             "SKCM": r"melanoma", "CESC": r"cervical", "MM": r"myeloma", "UCEC": r"endometri|uterine",
             "OV": r"ovar", "GBM": r"glioblastoma", "ccRCC": r"renal cell"}
    return {
        "htan_mentions": len(re.findall(r"\bHTAN\b|Human Tumou?r Atlas Network", t)),
        "wustl_mentions": len(re.findall(r"WUSTL|Washington University", t)),
        "cancer_type_codes_found": [c for c, name in types.items()
                                    if re.search(rf"\b{c}\b", t) or re.search(name, t, re.I)],
        "snatac_mentions": len(re.findall(r"snATAC|scATAC|snMultiome|multiome", t, re.I)),
        "htan_sample_id_like": sorted(set(re.findall(r"\b(?:HT|CE|VF|ML|CPT)\d{3,4}[A-Z0-9]*-[A-Za-z0-9]+\b", t)))[:20],
    }


def fingerprint_ok(fp: dict) -> bool:
    """Dataset fingerprint: HTAN + WUSTL atlas (or HTAN-style sample IDs), >=9 of the 11 cancer types, snATAC."""
    return (fp["htan_mentions"] >= 3 and (fp["wustl_mentions"] > 0 or bool(fp["htan_sample_id_like"]))
            and len(fp["cancer_type_codes_found"]) >= 9 and fp["snatac_mentions"] > 0)


def section_text(xml: str, title_regex: str) -> str:
    m = re.search(rf"<(sec|notes)\b[^>]*>\s*<title>\s*{title_regex}\s*</title>(.*?)</\1>", xml, re.S | re.I)
    return m.group(2) if m else ""


def strip_tags(x: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", x))).strip()


def github_refs(text: str) -> dict[str, str]:
    """owner/repo (or owner/) -> 400-char context snippet."""
    out: dict[str, str] = {}
    plain = strip_tags(text)
    for m in re.finditer(r"github\.com/([A-Za-z0-9_.-]+)(?:/([A-Za-z0-9_.-]+))?", plain):
        owner, repo = m.group(1), (m.group(2) or "").removesuffix(".git").rstrip(".")
        key = f"{owner}/{repo}" if repo else f"{owner}/"
        out.setdefault(key, plain[max(0, m.start() - 300): m.end() + 200])
    return out


def heuristic_repo_decision(repo: str, ref: dict, readme: str, desc: str, sources: list[dict],
                            author_orgs: list[str]) -> tuple[str, str]:
    """Deterministic fallback when Claude is unavailable."""
    text = f"{repo} {desc} {readme}".lower()
    via = " ".join(sorted(ref["found_via"]))
    ctx = (ref.get("context") or "").lower()
    mentions_source = any(
        (p["doi"] and (p["doi"] in text or (doi_suffix_pattern(p["doi"]) or "@@") in text)) or
        (len(p["title"]) > 30 and title_key(p["title"]) in title_key(text)) for p in sources)
    dataset_terms = bool(re.search(r"\bhtan\b|snatac|scatac|multiome|pan-?cancer|pancan", text))
    sample_ids = bool(re.search(r"\b(?:HT|CE|VF|ML|CPT)\d{3,4}[A-Z0-9]*-[A-Za-z0-9]+\b", f"{desc} {readme}"))
    in_paper = "paper full text" in via
    owner = repo.split("/")[0]
    if in_paper and re.search(r"reproduc|figures|publication", ctx + " " + repo.lower()):
        return "study_code", "cited in the paper's code-availability text as the analysis/figure code"
    if mentions_source and (dataset_terms or in_paper):
        return ("study_code" if owner in author_orgs else "related_study_code"), \
            "README/description cites the source paper and uses dataset terms"
    if owner in author_orgs and re.search(r"publication|paper", repo.lower()) and (
            sample_ids or re.search(r"(?<![a-z])(htan|cptac|snatac|scatac|multiome|epigen\w*|single.?nucle\w*)(?![a-z])", text)):
        return "related_study_code", ("author-org publication repo mentioning HTAN/CPTAC/snATAC or HTAN-style "
                                      "sample IDs (may hold annotations for overlapping biospecimens)")
    return "generic_tool", "no source citation or HTAN dataset markers found"


class GitHub:
    API = "https://api.github.com"

    def __init__(self, http: Http):
        self.http = http
        tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
        src = "env" if tok else ""
        if not tok:
            try:
                r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
                if r.returncode == 0 and r.stdout.strip():
                    tok, src = r.stdout.strip(), "gh auth token"
            except (OSError, subprocess.SubprocessError):
                pass
        self.auth = src or "unauthenticated"
        self.headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if tok:
            self.headers["Authorization"] = f"Bearer {tok}"

    def get(self, path: str, params: Optional[dict] = None) -> Any:
        return self.http.request(f"{self.API}{path}", params=params, headers=self.headers)

    def repo(self, full: str) -> Optional[dict]:
        return self.get(f"/repos/{full}")

    def org_repos(self, owner: str) -> list[dict]:
        out, page = [], 1
        kind = "orgs"
        while True:
            d = self.get(f"/{kind}/{owner}/repos", {"per_page": 100, "page": page, "type": "all"})
            if d is None and kind == "orgs" and page == 1:
                kind = "users"
                continue
            if not d:
                break
            out.extend(d)
            if len(d) < 100:
                break
            page += 1
        return out

    def forks_recursive(self, full: str, seen: Optional[set] = None, depth: int = 0) -> list[dict]:
        seen = seen if seen is not None else {full.lower()}
        out = []
        page = 1
        while True:
            d = self.get(f"/repos/{full}/forks", {"per_page": 100, "page": page, "sort": "oldest"})
            if not d:
                break
            for f in d:
                fn = f["full_name"]
                if fn.lower() in seen:
                    continue
                seen.add(fn.lower())
                out.append({"full_name": fn, "parent": full, "depth": depth + 1, "html_url": f["html_url"]})
                if f.get("forks_count", 0) > 0:
                    out.extend(self.forks_recursive(fn, seen, depth + 1))
            if len(d) < 100:
                break
            page += 1
        return out

    def search_repos(self, q: str) -> list[dict]:
        d = self.get("/search/repositories", {"q": q, "per_page": 50})
        return (d or {}).get("items", [])

    def readme(self, full: str, branch: str) -> str:
        for name in ("README.md", "README.rst", "README", "readme.md"):
            t = self.http.request(f"https://raw.githubusercontent.com/{full}/{branch}/{name}", as_json=False)
            if t:
                return t[:3000]
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-description", default=DEFAULT_DATASET_DESCRIPTION,
                    help="Free-text description of the evaluation dataset (used by Claude to propose source papers).")
    ap.add_argument("--source-doi", action="append", default=[],
                    help="Known source-paper DOI(s); added to (or, with --no-llm, replace) Claude's candidates.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--provenance", type=Path, default=DEFAULT_PROV)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--refresh", action="store_true", help="Ignore cached API responses (Claude cache is kept).")
    ap.add_argument("--max-citing", type=int, default=None, help="Cap citing papers per source (quick tests).")
    ap.add_argument("--citing", choices=["all", "reuse-only", "none"], default="all",
                    help="Which citing papers to blacklist (reuse-only requires --flag-reuse). Default: all.")
    ap.add_argument("--flag-reuse", action="store_true",
                    help="Use Claude (batched) to flag citing papers that re-use the dataset.")
    ap.add_argument("--reuse-batch-size", type=int, default=60)
    ap.add_argument("--citing-of-companions", action="store_true",
                    help="Also collect papers citing companion papers (default: only the source + its preprints).")
    ap.add_argument("--include-tool-repos", action="store_true",
                    help="Also blacklist the authors' generic tool repos cited by the paper (pipelines etc.).")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "max"])
    ap.add_argument("--no-llm", action="store_true", help="Disable Claude.")
    ap.add_argument("--fallback-query", default="HTAN single-nucleus ATAC multiome 11 cancer types",
                    help="OpenAlex search used when no candidate could be verified (e.g. LLM unavailable).")
    ap.add_argument("--min-title-len", type=int, default=30, help="Minimum length for title_contains entries.")
    ap.add_argument("--openalex-api-key", default=os.environ.get("OPENALEX_API_KEY", ""))
    ap.add_argument("--s2-api-key", default=os.environ.get("S2_API_KEY", ""))
    ap.add_argument("--mailto", default="", help="Optional OpenAlex polite-pool email (default: none sent).")
    ap.add_argument("--max-wait", type=float, default=900.0, help="Max seconds to wait on a rate limit.")
    args = ap.parse_args()

    if args.citing == "reuse-only" and not args.flag_reuse:
        ap.error("--citing reuse-only requires --flag-reuse")

    http = Http(args.cache_dir, refresh=args.refresh, max_wait=args.max_wait)
    sch = Scholarly(http, args.openalex_api_key, args.mailto, args.s2_api_key)
    claude = Claude(args.model, args.cache_dir, enabled=not args.no_llm, effort=args.effort)
    gh = GitHub(http)
    bl = Blacklist(args.min_title_len)
    prov: dict[str, Any] = {
        "generator": "scripts/make_literature_blacklist.py",
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                   if k not in {"openalex_api_key", "s2_api_key", "mailto", "cache_dir", "out", "provenance"}},
        "github_auth": gh.auth,
    }
    log(f"cache: {args.cache_dir}   github auth: {gh.auth}   llm: {'off' if args.no_llm else args.model}")

    # ---------------- 1. Source papers ----------------
    log("[1/5] identifying source papers")
    cands = [{"title": "", "doi": d, "role": "source", "rationale": "--source-doi", "first_author": "",
              "year": 0, "venue": ""} for d in args.source_doi]
    llm_sugg = claude.ask_json(
        "propose_sources", SYSTEM,
        "Dataset used in an evaluation:\n" + args.dataset_description + "\n\n"
        "1) List the publication(s) that generated / first described this exact dataset (role=source), any "
        "preprint versions (role=preprint), data-descriptor papers (role=data_descriptor), and companion papers "
        "by the same consortium/lab that analyse the SAME patients/biospecimens and could reveal their "
        "annotations (role=companion). Give exact titles; DOI only if certain.\n"
        "2) code_repos: GitHub repos (owner/repo) holding the analysis code or cell annotations for these papers.\n"
        "3) data_resources: data portal pages / accessions (GEO, dbGaP, Synapse, Zenodo, figshare, CELLxGENE, "
        "HTAN portal) holding per-cell annotations or processed objects for this dataset.",
        CANDIDATE_SCHEMA) or {}
    cands += llm_sugg.get("papers", [])
    prov["llm_suggestions"] = llm_sugg

    verified: list[dict] = []
    rejected: list[dict] = []
    for c in cands:
        w, ev = verify_candidate(sch, c) if (c.get("title") or c.get("doi")) else (None, "empty")
        if c.get("rationale") == "--source-doi" and not w and c.get("doi"):
            w = sch.oa_by_doi(norm_doi(c["doi"]))
            ev = "OpenAlex DOI lookup (user-supplied)" if w else "DOI not found in OpenAlex"
        if not w:
            rejected.append({**c, "evidence": ev})
            continue
        p = paper_from_openalex(w)
        p.update(role=c["role"], evidence=ev, llm_rationale=c.get("rationale"))
        verified.append(p)
    # de-duplicate verified (same work suggested twice); keep strongest role
    role_rank = {"source": 0, "preprint": 1, "data_descriptor": 2, "companion": 3}
    byid: dict[str, dict] = {}
    for p in sorted(verified, key=lambda x: role_rank[x["role"]]):
        byid.setdefault(p["openalex"], p)
    verified = sorted(byid.values(), key=lambda x: (role_rank[x["role"]], x["title"]))
    sources = [p for p in verified if p["role"] == "source"]
    if not sources:
        # No (verified) LLM/CLI candidate: search OpenAlex and verify by full-text fingerprint. Among the
        # passing works, the source is the one that does not cite another passing work (re-users cite it).
        log(f"  no verified candidate ({claude.disabled_reason or 'LLM gave none'}); "
            f"falling back to OpenAlex search: {args.fallback_query!r}")
        passing = []
        for w in sch.oa_search(args.fallback_query, n=10):
            p = paper_from_openalex(w)
            p["referenced_works"] = w.get("referenced_works") or []
            ft = sch.pmc_fulltext(p["pmcid"]) if p.get("pmcid") else ""
            fp = fingerprint_fulltext(ft) if ft else None
            if fp and fingerprint_ok(fp):
                p["_ft"] = ft.lower()
                passing.append(p)
        pids = {p["openalex"] for p in passing}
        for p in passing:
            cites_other = sorted(pids & set(p["referenced_works"]))
            # Also catch references OpenAlex missed: other candidate's DOI or title in this paper's full text.
            cites_other += sorted(o["openalex"] for o in passing if o is not p and o["openalex"] not in cites_other
                                  and ((o["doi"] and o["doi"] in p["_ft"]) or
                                       (len(o["title"]) > 30 and title_key(o["title"]) in title_key(p["_ft"]))))
            p.pop("_ft", None)
            if not cites_other:
                p.update(role="source", evidence="OpenAlex search + PMC full-text fingerprint; cites no other "
                         "fingerprint-matching candidate")
                verified.append(p)
            else:
                rejected.append({"title": p["title"], "doi": p["doi"], "role": "source?",
                                 "evidence": f"fingerprint matches but cites {cites_other} (likely re-user)"})
        sources = [p for p in verified if p["role"] == "source"]
    if not sources:
        log("ERROR: no source paper could be verified"); return 2

    # Full-text fingerprint check of the source(s); also harvest code/data links.
    fulltexts: dict[str, str] = {}
    for p in sources:
        ft = sch.pmc_fulltext(p["pmcid"]) if p.get("pmcid") else ""
        fulltexts[p["openalex"]] = ft
        fp = fingerprint_fulltext(ft or p.get("abstract", ""))
        p["fulltext_fingerprint"] = fp
        p["fulltext_source"] = "NCBI efetch db=pmc" if ft else "OpenAlex abstract only"
        ok = fingerprint_ok(fp)
        p["dataset_match"] = ok
        log(f"  source: {p['title']} doi={p['doi']} pmid={p['pmid']} pmcid={p['pmcid']} "
            f"fingerprint_ok={ok} ({fp['htan_mentions']} HTAN mentions, {len(fp['cancer_type_codes_found'])}/11 types)")
        s2 = sch.s2_paper(p["doi"]) if p["doi"] else None
        if s2:
            p["s2"] = s2.get("paperId")
            p["apis"].add("semantic_scholar")
            ext = s2.get("externalIds") or {}
            p["cross_check"] = {"s2_pmid": ext.get("PubMed"), "s2_pmcid": ext.get("PubMedCentral"),
                                "s2_title_sim": round(title_sim(s2.get("title"), p["title"]), 3)}
    for p in verified:
        if p["role"] != "source":
            log(f"  {p['role']}: {p['title']} doi={p['doi']} ({p['evidence']})")
    for r in rejected:
        log(f"  rejected candidate: {r.get('title') or r.get('doi')} -> {r['evidence']}")

    # Preprints of source + companions
    log("[2/5] searching preprints")
    preprints: list[dict] = []
    for p in [x for x in verified if x["role"] in ("source", "companion", "data_descriptor")]:
        found: list[tuple[dict, str]] = []
        if p["doi"]:
            for c in sch.biorxiv_pubs(p["doi"]):
                d = norm_doi(c.get("preprint_doi"))
                if d:
                    w = sch.oa_by_doi(d)
                    found.append((paper_from_openalex(w) if w else {"doi": d, "title": c.get("preprint_title"), "apis": set()},
                                  f"bioRxiv pubs API ({c['_server']})"))
        for it in sch.crossref_posted(p["title"]):
            t = (it.get("title") or [""])[0]
            if title_sim(t, p["title"]) >= 0.85:
                found.append(({"doi": norm_doi(it.get("DOI")), "title": t, "urls": [it.get("URL")] if it.get("URL") else [],
                               "apis": {"crossref"}}, "Crossref posted-content title match"))
        for w in sch.oa_title_search(p["title"]):
            if w.get("id") != p["openalex"] and title_sim(w.get("title"), p["title"]) >= 0.85:
                found.append((paper_from_openalex(w), "OpenAlex title search (other version)"))
        s2m = sch.s2_match(p["title"])
        if s2m and title_sim(s2m.get("title"), p["title"]) >= 0.85:
            q = paper_from_s2(s2m)
            if q["doi"] and q["doi"] != p["doi"]:
                found.append((q, "Semantic Scholar title match (other DOI)"))
        # Preprint hosts listed as locations of the same OpenAlex work
        for u in p["urls"]:
            if PREPRINT_HOST_RE.search(u):
                found.append(({"doi": "", "title": p["title"], "urls": [u], "apis": {"openalex"}},
                              "OpenAlex location on preprint server"))
        for q, how in found:
            if q.get("doi") == p["doi"] and not q.get("urls"):
                continue
            q = {**q, "apis": set(q.get("apis") or []), "role": "preprint", "preprint_of": p["doi"], "evidence": how}
            preprints.append(q)
    preprints = merge_papers(preprints)
    for q in preprints:
        log(f"  preprint/version: {q.get('doi') or q.get('urls')} ({q.get('evidence')})")

    for p in verified:
        cat = {"source": "source", "preprint": "preprint", "companion": "companion", "data_descriptor": "data_descriptor"}[p["role"]]
        bl.add_paper(p, cat, f"{cat} paper: {p['title']}", strong=True)
    for q in preprints:
        bl.add_paper(q, "preprint", f"preprint/version of {q.get('preprint_of')}", strong=True)

    # ---------------- 2. Citing papers ----------------
    log("[3/5] collecting citing papers")
    # Preprints count as citation targets only when the paper they are a version of is one:
    # OpenAlex merges a preprint with its published version, so a companion's preprint would
    # otherwise pull in all of the companion's citing papers.
    target_roles = {"source", "preprint"} | ({"companion", "data_descriptor"} if args.citing_of_companions else set())
    target_dois = {p.get("doi") for p in verified if p["role"] in target_roles}
    cite_targets = [p for p in verified if p["role"] in target_roles] + \
                   [q for q in preprints if q.get("openalex") and q.get("preprint_of") in target_dois]
    own = {p.get("doi") for p in verified + preprints if p.get("doi")}
    citing_raw: list[dict] = []
    per_target_counts = {}
    for t in cite_targets:
        c_oa = sch.oa_citing(t["openalex"], args.max_citing) if t.get("openalex") else []
        c_s2 = sch.s2_citing(t["doi"], args.max_citing) if t.get("doi") else []
        c_pm = sch.ncbi_citedin(t["pmid"]) if t.get("pmid") else []
        if args.max_citing:
            c_oa, c_s2, c_pm = c_oa[:args.max_citing], c_s2[:args.max_citing], c_pm[:args.max_citing]
        sums = sch.ncbi_pubmed_summaries(c_pm)
        recs = [paper_from_openalex(w) for w in c_oa] + [paper_from_s2(p) for p in c_s2] + \
               [paper_from_pubmed(u, sums[u]) if u in sums else {"pmid": u, "apis": {"ncbi_pubmed"}} for u in c_pm]
        for r in recs:
            r["cites"] = t.get("doi") or t.get("openalex")
        per_target_counts[t.get("doi") or t.get("openalex")] = {"openalex": len(c_oa), "semantic_scholar": len(c_s2),
                                                                 "ncbi_citedin": len(c_pm)}
        citing_raw.extend(recs)
        log(f"  {t.get('doi')}: OpenAlex {len(c_oa)}, Semantic Scholar {len(c_s2)}, NCBI cited-in {len(c_pm)}")
    citing = merge_papers(citing_raw)
    citing = [c for c in citing if c.get("doi") not in own]
    citing.sort(key=lambda c: (c.get("doi") or "~", c.get("pmid") or "", title_key(c.get("title"))))
    if args.max_citing:
        citing = citing[: args.max_citing * max(1, len(cite_targets))]
    # Fill PMIDs / PMCIDs via the PMC ID converter
    need_doi = sorted({c["doi"] for c in citing if c.get("doi") and (not c.get("pmid") or not c.get("pmcid"))})
    need_pmid = sorted({c["pmid"] for c in citing if c.get("pmid") and (not c.get("pmcid") or not c.get("doi"))},
                       key=int)
    conv = {**sch.ncbi_ids_for_dois(need_doi), **sch.ncbi_ids_for_pmids(need_pmid)}
    for c in citing:
        m = conv.get(c.get("doi") or "") or conv.get(c.get("pmid") or "")
        if m:
            for f in ("pmid", "pmcid", "doi"):
                if not c.get(f) and m.get(f):
                    c[f] = m[f]
                    c["apis"].add("ncbi_eutils_idmap")
    log(f"  merged citing papers: {len(citing)}")

    # Optional: Claude flags data re-use (batched)
    reuse: dict[int, dict] = {}
    if args.flag_reuse and claude.enabled:
        src_desc = "; ".join(f"{p['title']} (doi:{p['doi']})" for p in sources)
        for i in range(0, len(citing), args.reuse_batch_size):
            batch = citing[i:i + args.reuse_batch_size]
            lines = [json.dumps({"id": str(i + j), "title": c.get("title"), "abstract": (c.get("abstract") or "")[:1500]})
                     for j, c in enumerate(batch)]
            out = claude.ask_json(
                "flag_reuse", SYSTEM,
                f"Source study: {src_desc}\nDataset: {args.dataset_description}\n\n"
                "For each citing paper below (JSON lines), decide whether it re-uses / re-analyses the source "
                "study's data (the HTAN WUSTL multiome samples, its annotations, peaks or derived objects) rather "
                "than merely citing it. reuse = yes | likely | no | unclear.\n\n" + "\n".join(lines),
                REUSE_SCHEMA) or {}
            for r in out.get("papers", []):
                if r.get("id", "").isdigit():
                    reuse[int(r["id"])] = r
            log(f"  reuse-flagged batch {i // args.reuse_batch_size + 1}: {len(out.get('papers', []))} judged")
    for idx, c in enumerate(citing):
        if idx in reuse:
            c["reuse"] = reuse[idx]["reuse"]
            c["reuse_reason"] = reuse[idx]["reason"]
        include = args.citing == "all" or (args.citing == "reuse-only" and c.get("reuse") in ("yes", "likely"))
        c["blacklisted"] = include
        if include:
            bl.add_paper(c, "citing_paper", f"cites {c.get('cites')}" + (f"; reuse={c['reuse']}" if c.get("reuse") else ""))

    # ---------------- 3. Code repositories ----------------
    log(f"[4/5] code repositories (GitHub auth: {gh.auth})")
    refs: dict[str, dict] = {}
    for p in sources:
        for k, ctx in github_refs(fulltexts.get(p["openalex"], "") + " " + p.get("abstract", "")).items():
            refs.setdefault(k, {"context": ctx, "found_via": set()})["found_via"].add("paper full text (PMC)")
    for r in llm_sugg.get("code_repos", []):
        m = re.search(r"(?:github\.com/)?([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", r)
        if m:
            k = f"{m.group(1)}/{m.group(2).removesuffix('.git')}"
            refs.setdefault(k, {"context": "", "found_via": set()})["found_via"].add("Claude suggestion")
    for p in sources:
        for q in (f'"{p["doi"]}"', f'"{doi_suffix_pattern(p["doi"]) or p["doi"]}" in:readme,description'):
            for it in gh.search_repos(q):
                refs.setdefault(it["full_name"], {"context": it.get("description") or "", "found_via": set()})[
                    "found_via"].add(f"GitHub repo search {q}")
    # Authoring orgs: owners of repos explicitly cited by the paper.
    owners = sorted({k.split("/")[0] for k, v in refs.items() if "paper full text (PMC)" in v["found_via"]} -
                    {"FelixKrueger", "samtools", "timoast", "broadinstitute"})
    owner_counts = defaultdict(int)
    for k, v in refs.items():
        if "paper full text (PMC)" in v["found_via"]:
            owner_counts[k.split("/")[0]] += 1
    author_orgs = [o for o in owners if owner_counts[o] >= 2 or f"{o}/" in refs]
    repo_meta: dict[str, dict] = {}
    for o in author_orgs:
        for r in gh.org_repos(o):
            s = f"{r['name']} {r.get('description') or ''}".lower()
            repo_meta[r["full_name"]] = r
            if re.search(REPO_KEYWORDS, s):
                refs.setdefault(r["full_name"], {"context": r.get("description") or "", "found_via": set()})[
                    "found_via"].add(f"author org listing ({o}) keyword match")
    prov["author_orgs"] = author_orgs
    candidates = sorted(k for k in refs if not k.endswith("/"))
    info: dict[str, dict] = {}
    for k in candidates:
        meta = repo_meta.get(k) or gh.repo(k)
        if not meta:
            info[k] = {"exists": False}
            continue
        info[k] = {"exists": True, "full_name": meta["full_name"], "description": meta.get("description"),
                   "forks_count": meta.get("forks_count", 0), "default_branch": meta.get("default_branch", "main"),
                   "fork": meta.get("fork", False)}
    existing = [k for k in candidates if info[k]["exists"]]
    repo_lines = []
    readmes: dict[str, str] = {}
    for k in existing:
        rd = gh.readme(info[k]["full_name"], info[k]["default_branch"])
        readmes[k] = rd
        repo_lines.append(json.dumps({"repo": info[k]["full_name"], "description": info[k]["description"],
                                      "found_via": sorted(refs[k]["found_via"]),
                                      "context_in_paper": refs[k]["context"][:500], "readme_head": rd[:1200]}))
    decisions: dict[str, dict] = {}
    if existing:
        out = claude.ask_json(
            "classify_repos", SYSTEM,
            "Source study: " + "; ".join(f"{p['title']} (doi:{p['doi']})" for p in sources) +
            "\nCompanion papers: " + "; ".join(p["title"] for p in verified if p["role"] == "companion") +
            "\nDataset: " + args.dataset_description +
            "\n\nClassify each GitHub repo: study_code = analysis/annotation code or data files of the source study; "
            "related_study_code = code/data of another publication analysing overlapping HTAN WUSTL / CPTAC samples "
            "(could leak cell-type, cancer-type or program annotations for these biospecimens); generic_tool = a "
            "reusable pipeline/tool with no dataset-specific annotations; unrelated.\n\n" + "\n".join(repo_lines),
            REPO_SCHEMA) or {}
        for d in out.get("repos", []):
            decisions[d["repo"].lower()] = d
    repos_out = []
    for k in candidates:
        i = info[k]
        rec = {"repo": k, **i, "found_via": sorted(refs[k]["found_via"])}
        if not i["exists"]:
            rec["decision"] = "not_found"
            repos_out.append(rec)
            continue
        d = decisions.get(i["full_name"].lower())
        if d:
            rec["decision"], rec["reason"], rec["decided_by"] = d["decision"], d["reason"], "claude"
        else:  # heuristic fallback (no LLM / missing decision)
            rec["decision"], rec["reason"] = heuristic_repo_decision(
                k, refs[k], readmes.get(k, ""), i.get("description") or "", sources, author_orgs)
            rec["decided_by"] = "heuristic"
        include = rec["decision"] in ("study_code", "related_study_code") or \
            (args.include_tool_repos and rec["decision"] == "generic_tool" and k.split("/")[0] in author_orgs)
        rec["blacklisted"] = include
        if include:
            fn = i["full_name"]
            names = {fn, k}
            forks = gh.forks_recursive(fn) if i.get("forks_count", 0) > 0 else []
            rec["forks"] = forks
            for n in sorted(names):
                for pat in (f"github.com/{n}", f"raw.githubusercontent.com/{n}", f"api.github.com/repos/{n}",
                            f"codeload.github.com/{n}"):
                    bl.add("urls", pat.lower(), "repo", f"{rec['decision']}: {rec.get('reason', '')}"[:300],
                           ["github"] + (["claude"] if rec.get("decided_by") == "claude" else []))
            for f in forks:
                for pat in (f"github.com/{f['full_name']}", f"raw.githubusercontent.com/{f['full_name']}"):
                    bl.add("urls", pat.lower(), "fork", f"fork (depth {f['depth']}) of {f['parent']}", ["github"])
        repos_out.append(rec)
        n_forks = len(rec.get("forks", []))
        log(f"  repo {k}: {rec['decision']}" + (f" -> BLACKLISTED (+{n_forks} forks)" if include else ""))
    prov["repos"] = repos_out

    # ---------------- 4. Data portals ----------------
    log("[5/5] data portals / accessions")
    data_res: list[dict] = []

    def add_data(pattern: str, kind: str, how: str, verified_ok: bool, apis: list[str]) -> None:
        data_res.append({"pattern": pattern, "kind": kind, "found_via": how, "verified": verified_ok})
        if verified_ok:
            bl.add("urls", pattern.lower(), "data_portal", f"{kind} ({how})", apis)

    for p in sources:
        xml = fulltexts.get(p["openalex"], "")
        da = strip_tags(section_text(xml, r"Data availability"))
        prov.setdefault("data_availability_text", {})[p["doi"]] = da
        for acc in sorted(set(re.findall(r"\bGSE\d{4,}\b", da))):
            add_data(acc, "GEO series", "Data availability section", sch.ncbi_gds_exists(acc), ["ncbi_gds"])
        for acc in sorted(set(re.findall(r"\bphs\d{6}\b", da))):
            add_data(acc, "dbGaP study", "Data availability section", True, ["pmc_fulltext"])
        for acc in sorted(set(re.findall(r"\bsyn\d{6,}\b", da))):
            ok = http.request(f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{acc}/type") is not None
            add_data(acc, "Synapse entity", "Data availability section", ok, ["synapse"])
        for u in sorted(set(re.findall(r"https?://[^\s)\]>,;]+", da))):
            nu = norm_url(u)
            if any(h in nu for h in ("humantumoratlas.org", "zenodo.org", "figshare", "synapse.org", "cellxgene",
                                      "data.mendeley.com", "ega-archive")):
                add_data(nu.split("/")[0] if nu.count("/") <= 1 or "humantumoratlas" in nu else nu,
                         "data portal", "Data availability section URL", True, ["pmc_fulltext"])
        for z in re.findall(r"10\.5281/zenodo\.\d+", da):
            add_data(z.replace("10.5281/", ""), "Zenodo record", "Data availability section", True, ["pmc_fulltext"])
            bl.add("dois", z, "data_portal", "Zenodo DOI in data availability", ["pmc_fulltext"])
        if p.get("pmid"):
            for acc in sch.ncbi_gds_links(p["pmid"]):
                add_data(acc, "GEO series", "NCBI elink pubmed->gds", True, ["ncbi_elink"])
    # CELLxGENE collections whose DOI is a verified paper
    vdois = {p["doi"] for p in verified if p.get("doi")}
    cxg = http.request("https://api.cellxgene.cziscience.com/curation/v1/collections") or []
    for c in cxg:
        if norm_doi(c.get("doi")) in vdois:
            add_data(norm_url(c["collection_url"]), "CELLxGENE collection", f"curation API doi={c.get('doi')}", True,
                     ["cellxgene"])
            for ds in c.get("datasets") or []:
                if ds.get("dataset_id"):
                    bl.add("urls", ds["dataset_id"], "data_portal", "CELLxGENE dataset id", ["cellxgene"])
    # Zenodo records referencing the DOIs
    for d in sorted(vdois):
        z = http.request("https://zenodo.org/api/records", params={"q": f'"{d}"', "size": 25})
        for h in ((z or {}).get("hits") or {}).get("hits", []):
            blob = json.dumps(h.get("metadata", {})).lower()
            if d in blob:
                add_data(f"zenodo.org/records/{h['id']}", "Zenodo record", f"Zenodo search references {d}", True,
                         ["zenodo"])
                if h.get("doi"):
                    bl.add("dois", norm_doi(h["doi"]), "data_portal", f"Zenodo record referencing {d}", ["zenodo"])
    # Claude-suggested resources: only accepted if they verify via an API.
    for r in llm_sugg.get("data_resources", []):
        for acc in re.findall(r"\bGSE\d{4,}\b", r):
            add_data(acc, "GEO series", f"Claude suggestion: {r[:80]}", sch.ncbi_gds_exists(acc), ["claude", "ncbi_gds"])
        for acc in re.findall(r"\bsyn\d{6,}\b", r):
            e = http.request(f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{acc}/type")
            add_data(acc, "Synapse entity", f"Claude suggestion: {r[:80]}", e is not None, ["claude", "synapse"])
        if not re.search(r"\b(GSE\d+|syn\d+)\b", r):
            data_res.append({"pattern": r, "kind": "suggestion", "found_via": "Claude suggestion",
                             "verified": False, "note": "not auto-verifiable; not added"})
    # dedupe data_res records
    seen_dr, dr = set(), []
    for d in data_res:
        key = (d["pattern"].lower(), d["kind"], d["found_via"])
        if key not in seen_dr:
            seen_dr.add(key)
            dr.append(d)
    prov["data_resources"] = dr
    for d in dr:
        if d["verified"]:
            log(f"  data: {d['pattern']} ({d['kind']}; {d['found_via']})")

    # ---------------- Write outputs ----------------
    out = bl.to_yaml_dict()
    def jsonable(p: dict) -> dict:
        q = {k: (sorted(v) if isinstance(v, set) else v) for k, v in p.items() if k != "abstract"}
        return q
    prov["papers"] = {
        "verified_candidates": [jsonable(p) for p in verified],
        "rejected_candidates": rejected,
        "preprints": [jsonable(q) for q in preprints],
        "citing_counts_per_target": per_target_counts,
        "citing": [jsonable(c) for c in citing],
    }
    prov["summary"] = {
        "sources": [p["doi"] for p in sources],
        "companions": [p["doi"] for p in verified if p["role"] == "companion"],
        "preprints": len(preprints),
        "citing_papers": len(citing),
        "citing_blacklisted": sum(1 for c in citing if c.get("blacklisted")),
        "reuse_flagged_yes_or_likely": sum(1 for c in citing if c.get("reuse") in ("yes", "likely")),
        "repos_blacklisted": sum(1 for r in repos_out if r.get("blacklisted")),
        "forks_blacklisted": sum(len(r.get("forks", [])) for r in repos_out if r.get("blacklisted")),
        "data_resources_blacklisted": sum(1 for d in dr if d["verified"]),
        "yaml_counts": {k: len(v) for k, v in out.items()},
        "title_contains_skipped_too_short": bl.skipped_titles,
    }
    prov["entries"] = {k: {v: bl.entries[k][v] for v in out[k]} for k in out}
    prov["llm_calls"] = claude.calls
    prov["llm_disabled_reason"] = claude.disabled_reason

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# GENERATED FILE - do not edit by hand.",
        "# Literature/code blacklist for the HTAN WUSTL multiome evaluation tasks.",
        "# Regenerate with:",
        "#   .venv/bin/python scripts/make_literature_blacklist.py"
        + (f" --citing {args.citing}" if args.citing != "all" else "")
        + (" --flag-reuse" if args.flag_reuse else ""),
        "# Provenance (why each entry is here, and from which API): "
        + os.path.relpath(args.provenance, args.out.parent),
        "# Source paper(s): " + "; ".join(f"{p['title']} (doi:{p['doi']}, PMID {p['pmid']})" for p in sources),
        f"# Counts: {json.dumps(prov['summary']['yaml_counts'])}",
        "# Load with: literature_blacklist: <this mapping>  (keys accepted by _normalize_literature_blacklist)",
        "",
    ]
    body = yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=1000, default_flow_style=False)
    footer = []
    flagged = [c for c in citing if c.get("reuse") in ("yes", "likely")]
    if flagged:
        footer = ["", "# --- Citing papers flagged by Claude as (likely) re-using the dataset (informational) ---"]
        footer += [f"# [{c['reuse']}] {c.get('doi') or c.get('pmid')}: {c.get('title', '')[:150]}" for c in flagged]
    args.out.write_text("\n".join(header) + body + "\n".join(footer) + ("\n" if footer else ""))
    args.provenance.write_text(json.dumps(prov, indent=1, sort_keys=False, default=lambda o: sorted(o) if isinstance(o, set) else str(o)) + "\n")
    log(f"wrote {args.out} ({prov['summary']['yaml_counts']}) and {args.provenance}")
    log(f"http: {dict(sorted(http.stats.items()))}")
    log(f"summary: {json.dumps({k: v for k, v in prov['summary'].items() if k != 'title_contains_skipped_too_short'})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
