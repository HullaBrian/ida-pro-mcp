"""CAPA integration — read capability findings from the CAPA Explorer IDA plugin.

This module exposes cached CAPA results that are already stored in the IDB by
the CAPA Explorer plugin (Edit > Plugins > CAPA Explorer).  It never re-runs
analysis; if results have not been generated yet it tells the caller to run the
plugin first.

Enable these tools by connecting with ?ext=capa in the MCP URL, e.g.:
    http://127.0.0.1:13337/mcp?ext=capa

Reading strategy (tried in order):
  1. capa.ida.helpers — if capa is installed in IDA's Python environment its
     own cache-read function handles every version difference automatically.
  2. Direct netnode read — parse the "$ com.mandiant.capa" IDB netnode blob
     (gzipped or plain JSON) without requiring capa to be importable.
"""

import gzip
import json
from typing import Annotated, NotRequired, Optional, TypedDict

from .rpc import tool, ext
from .sync import idasync, IDAError
from .utils import parse_address


# ---------------------------------------------------------------------------
# Storage constants
# ---------------------------------------------------------------------------

# CAPA netnode names across Mandiant / FireEye branding
_CAPA_NETNODE_NAMES = [
    "$ com.mandiant.capa",
    "$ com.fireeye.capa",
    "$ capa",
]

# Blob slot/tag pairs to try when reading the netnode directly.
# CAPA v7 uses (0, ord('A')); older versions may differ.
_BLOB_CANDIDATES: list[tuple[int, str]] = [
    (0, "A"),
    (0, "S"),
    (0, "D"),
    (0, "\x00"),
    (1, "A"),
    (1, "S"),
]


# ---------------------------------------------------------------------------
# TypedDicts for structured output
# ---------------------------------------------------------------------------


class CapaAttack(TypedDict):
    technique: str
    subtechnique: NotRequired[str]
    tactic: NotRequired[str]
    display: str


class CapaMbc(TypedDict):
    objective: NotRequired[str]
    behavior: str
    method: NotRequired[str]
    identifier: NotRequired[str]
    display: str


class CapaCapability(TypedDict):
    name: str
    namespace: NotRequired[str]
    description: NotRequired[str]
    attack: list[CapaAttack]
    mbc: list[CapaMbc]
    matched_addresses: list[str]


class CapaMetadata(TypedDict):
    timestamp: str
    capa_version: str
    binary_path: str
    binary_md5: str
    rules: list[str]
    arch: str
    os: str
    format: str


class CapaStatusResult(TypedDict):
    available: bool
    timestamp: NotRequired[str]
    capa_version: NotRequired[str]
    binary: NotRequired[str]
    capability_count: NotRequired[int]
    error: NotRequired[str]


class CapaCapabilitiesResult(TypedDict):
    meta: CapaMetadata
    capabilities: list[CapaCapability]
    total: int


class CapaFuncResult(TypedDict):
    addr: str
    capabilities: list[CapaCapability]
    count: int


# ---------------------------------------------------------------------------
# Result reading
# ---------------------------------------------------------------------------


def _decode_blob(blob: bytes) -> Optional[dict]:
    """Try to decode a netnode blob as plain or gzipped JSON."""
    if isinstance(blob, tuple):
        blob = blob[0]
    blob = bytes(blob)
    for decode in (_as_json, _as_gzip_json):
        result = decode(blob)
        if result and isinstance(result, dict) and "rules" in result:
            return result
    return None


def _as_json(blob: bytes) -> Optional[dict]:
    try:
        return json.loads(blob)
    except Exception:
        return None


def _as_gzip_json(blob: bytes) -> Optional[dict]:
    try:
        return json.loads(gzip.decompress(blob))
    except Exception:
        return None


def _try_capa_python_api() -> Optional[dict]:
    """Ask capa's own helpers to return the cached result document."""
    try:
        import capa.ida.helpers as _helpers  # type: ignore[import]
    except ImportError:
        return None

    for fn_name in (
        "get_capa_results",
        "get_capa_cached_results",
        "load_and_verify_cached_results",
        "load_capa_results",
    ):
        fn = getattr(_helpers, fn_name, None)
        if not callable(fn):
            continue
        try:
            result = fn()
        except Exception:
            continue
        if result is None:
            continue
        # Pydantic v2: mode='json' serialises datetime/UUID/etc. to plain types
        if hasattr(result, "model_dump"):
            try:
                return result.model_dump(mode="json")
            except TypeError:
                return result.model_dump()
        # Pydantic v1: .json() → str, then parse back to get JSON-safe types
        if hasattr(result, "dict"):
            return json.loads(result.json())
        if isinstance(result, dict):
            return result

    return None


def _try_netnode() -> Optional[dict]:
    """Read CAPA result blob directly from the IDB netnode."""
    try:
        import ida_netnode
    except ImportError:
        return None

    for name in _CAPA_NETNODE_NAMES:
        try:
            nd = ida_netnode.netnode(name, 0, False)
            if nd == ida_netnode.BADNODE:
                continue
            for slot, tag in _BLOB_CANDIDATES:
                blob = nd.getblob(slot, tag)
                if not blob:
                    continue
                result = _decode_blob(blob)
                if result is not None:
                    return result
        except Exception:
            continue

    return None


def _read_raw_results() -> Optional[dict]:
    result = _try_capa_python_api()
    if result is not None:
        return result
    return _try_netnode()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_addr(raw) -> str:
    """Normalise a CAPA match address (int or {type, value} dict) to hex."""
    if isinstance(raw, int):
        return hex(raw)
    if isinstance(raw, dict):
        val = raw.get("value") or raw.get("offset") or 0
        return hex(int(val))
    return "0x0"


def _parse_attack(entry) -> CapaAttack:
    if isinstance(entry, (list, tuple)):
        # Old format: [tactic, technique, subtechnique?, id]
        parts = [str(p) for p in entry if p]
        display = " :: ".join(parts)
        technique = parts[-1] if parts else ""
        return CapaAttack(technique=technique, display=display)

    # New format: dict with "parts", "technique", etc.
    parts = [str(p) for p in entry.get("parts", []) if p]
    technique = entry.get("technique", "")
    subtechnique = entry.get("subtechnique", "") or None
    tactic = entry.get("tactic", parts[0] if parts else "") or None
    display = " :: ".join(parts) if parts else technique
    result: CapaAttack = CapaAttack(technique=technique, display=display)
    if subtechnique:
        result["subtechnique"] = subtechnique
    if tactic:
        result["tactic"] = tactic
    return result


def _parse_mbc(entry) -> CapaMbc:
    if isinstance(entry, (list, tuple)):
        parts = [str(p) for p in entry if p]
        display = " :: ".join(parts)
        behavior = parts[1] if len(parts) > 1 else (parts[0] if parts else "")
        return CapaMbc(behavior=behavior, display=display)

    parts = [str(p) for p in entry.get("parts", []) if p]
    behavior = entry.get("behavior", parts[1] if len(parts) > 1 else "")
    objective = entry.get("objective", parts[0] if parts else "") or None
    method = entry.get("method") or None
    identifier = entry.get("id") or entry.get("identifier") or None
    display = " :: ".join(parts) if parts else behavior
    result: CapaMbc = CapaMbc(behavior=behavior, display=display)
    if objective:
        result["objective"] = objective
    if method:
        result["method"] = method
    if identifier:
        result["identifier"] = identifier
    return result


def _extract_capabilities(raw: dict) -> list[CapaCapability]:
    caps: list[CapaCapability] = []
    for rule_name, rule_data in raw.get("rules", {}).items():
        meta = rule_data.get("meta", {})
        name = meta.get("name", rule_name)
        namespace = meta.get("namespace") or None
        description = meta.get("description") or None

        attack = [_parse_attack(a) for a in meta.get("attack", [])]
        mbc = [_parse_mbc(m) for m in meta.get("mbc", [])]

        addrs: set[str] = set()
        for match_pair in rule_data.get("matches", []):
            if isinstance(match_pair, (list, tuple)) and match_pair:
                addrs.add(_extract_addr(match_pair[0]))

        cap: CapaCapability = CapaCapability(
            name=name,
            attack=attack,
            mbc=mbc,
            matched_addresses=sorted(addrs),
        )
        if namespace:
            cap["namespace"] = namespace
        if description:
            cap["description"] = description
        caps.append(cap)

    return caps


def _extract_meta(raw: dict) -> CapaMetadata:
    meta = raw.get("meta", {})
    sample = meta.get("sample", {})
    analysis = meta.get("analysis", {})

    binary_path = (
        sample.get("path")
        or sample.get("filename")
        or ""
    )
    binary_md5 = sample.get("md5", "")

    rules_raw = analysis.get("rules", [])
    if isinstance(rules_raw, str):
        rules_raw = [rules_raw]
    rules = [str(r) for r in rules_raw]

    ts_raw = meta.get("timestamp", "")
    timestamp = ts_raw.isoformat() if hasattr(ts_raw, "isoformat") else str(ts_raw) if ts_raw else ""

    return CapaMetadata(
        timestamp=timestamp,
        capa_version=str(meta.get("version", "")),
        binary_path=str(binary_path),
        binary_md5=str(binary_md5),
        rules=rules,
        arch=str(analysis.get("arch", "")),
        os=str(analysis.get("os", "")),
        format=str(analysis.get("format", "")),
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@ext("capa")
@tool
@idasync
def capa_status() -> CapaStatusResult:
    """Check whether CAPA capability analysis results are available in this IDB.

    CAPA results are generated once by the CAPA Explorer plugin
    (Edit > Plugins > CAPA Explorer) and then cached in the IDB — they do not
    need to be recomputed on every call.  This tool is a lightweight check that
    returns metadata without loading the full result set.

    If this returns available=false, open the CAPA Explorer plugin to run
    analysis first, then call this tool again.
    """
    raw = _read_raw_results()
    if raw is None:
        return CapaStatusResult(
            available=False,
            error=(
                "No CAPA results found in IDB. "
                "Run the CAPA Explorer plugin first "
                "(Edit > Plugins > CAPA Explorer), then retry."
            ),
        )

    meta = raw.get("meta", {})
    sample = meta.get("sample", {})
    n_rules = len(raw.get("rules", {}))

    result = CapaStatusResult(
        available=True,
        capability_count=n_rules,
    )
    ts = meta.get("timestamp")
    if ts:
        result["timestamp"] = ts
    version = meta.get("version")
    if version:
        result["capa_version"] = version
    binary = sample.get("path") or sample.get("filename")
    if binary:
        result["binary"] = binary

    return result


@ext("capa")
@tool
@idasync
def capa_capabilities(
    namespace: Annotated[
        str,
        "Filter by namespace prefix, e.g. 'anti-analysis', 'malware-category/ransomware'. Empty = all.",
    ] = "",
    attack_technique: Annotated[
        str,
        "Filter by ATT&CK technique ID, e.g. 'T1055'. Empty = all.",
    ] = "",
    mbc_objective: Annotated[
        str,
        "Filter by MBC objective keyword, e.g. 'Anti-Analysis'. Empty = all.",
    ] = "",
) -> CapaCapabilitiesResult:
    """Return CAPA capability findings cached in the current IDB.

    Reads results stored by the CAPA Explorer plugin — does NOT re-run
    analysis.  Each capability includes its name, namespace, ATT&CK and MBC
    mappings, and the binary addresses where the behavior was observed.

    Use capa_status to confirm results are available before calling this.
    Use capa_func_capabilities to query capabilities for a specific function.
    """
    raw = _read_raw_results()
    if raw is None:
        raise IDAError(
            "No CAPA results found in IDB. "
            "Run the CAPA Explorer plugin first (Edit > Plugins > CAPA Explorer)."
        )

    caps = _extract_capabilities(raw)

    if namespace:
        ns_lower = namespace.lower()
        caps = [c for c in caps if (c.get("namespace") or "").lower().startswith(ns_lower)]

    if attack_technique:
        tech_upper = attack_technique.upper()
        caps = [
            c for c in caps
            if any(tech_upper in a.get("technique", "").upper() for a in c.get("attack", []))
        ]

    if mbc_objective:
        obj_lower = mbc_objective.lower()
        caps = [
            c for c in caps
            if any(obj_lower in (m.get("objective") or "").lower() for m in c.get("mbc", []))
        ]

    return CapaCapabilitiesResult(
        meta=_extract_meta(raw),
        capabilities=caps,
        total=len(caps),
    )


@ext("capa")
@tool
@idasync
def capa_func_capabilities(
    addr: Annotated[
        str,
        "Function address or name to query (hex or symbol, e.g. '0x401000' or 'sub_401000').",
    ],
) -> CapaFuncResult:
    """Return CAPA capabilities matched at a specific function or address.

    Useful for understanding why a function is interesting — shows which
    malicious behaviors CAPA detected there.  The address is compared against
    all match locations in the cached result set; no re-analysis is performed.

    Tip: pair with list_funcs or lookup_funcs to iterate suspicious functions,
    then call this tool on each to get the CAPA context.
    """
    raw = _read_raw_results()
    if raw is None:
        raise IDAError(
            "No CAPA results found in IDB. "
            "Run the CAPA Explorer plugin first (Edit > Plugins > CAPA Explorer)."
        )

    ea = parse_address(addr)
    target = hex(ea)

    all_caps = _extract_capabilities(raw)
    matched = [c for c in all_caps if target in c.get("matched_addresses", [])]

    return CapaFuncResult(addr=target, capabilities=matched, count=len(matched))
