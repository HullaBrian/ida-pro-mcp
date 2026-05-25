"""Unicorn Engine emulation tools for IDA Pro MCP.

Emulates functions or instruction ranges using Unicorn Engine, with full
control over the initial register state.  Results include final register
values, memory reads, stop reason, and instruction count.

Enable these tools by connecting with ?ext=emu in the MCP URL, e.g.:
    http://127.0.0.1:13337/mcp?ext=emu

Requirements:
    pip install unicorn   (in IDA's Python environment)

The emulator maps all IDA segments into Unicorn before starting, so code and
data that are already analysed by IDA are available without any extra setup.
Unmapped accesses (e.g. stack, heap, external DLLs) are caught and reported
as the stop reason instead of crashing the emulation session.
"""

from typing import Annotated, NotRequired, Optional, TypedDict

from .rpc import tool, ext
from .sync import idasync, IDAError
from .utils import parse_address


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class EmuRegState(TypedDict):
    """Register state snapshot."""
    registers: dict[str, str]    # name → 0x-prefixed hex string


class EmuMemRead(TypedDict):
    """A memory range to read back after emulation."""
    address: str
    size: int
    data: str    # hex string of bytes


class EmuResult(TypedDict):
    start: str
    end: str
    arch: str
    stop_reason: str
    instructions_executed: int
    registers: dict[str, str]
    memory_reads: list[EmuMemRead]
    error: NotRequired[str]


# ---------------------------------------------------------------------------
# Unicorn helpers (self-contained, no uEmu plugin dependency)
# ---------------------------------------------------------------------------

_PAGE = 0x1000

def _align_down(x: int) -> int:
    return x & ~(_PAGE - 1)

def _align_up(x: int) -> int:
    return (x + _PAGE - 1) & ~(_PAGE - 1)


def _get_arch() -> str:
    """Detect IDA binary architecture; mirrors UEMU_HELPERS.get_arch()."""
    import idaapi
    ph = idaapi.ph
    PLFM_386  = idaapi.PLFM_386
    PLFM_ARM  = idaapi.PLFM_ARM
    PLFM_MIPS = idaapi.PLFM_MIPS
    PR_USE64  = idaapi.PR_USE64
    PR_USE32  = idaapi.PR_USE32

    def _is_be():
        if idaapi.IDA_SDK_VERSION >= 900:
            return idaapi.inf_is_be()
        return idaapi.cvar.inf.is_be()

    if ph.id == PLFM_386 and ph.flag & PR_USE64:
        return "x64"
    if ph.id == PLFM_386 and ph.flag & PR_USE32:
        return "x86"
    if ph.id == PLFM_ARM and ph.flag & PR_USE64:
        return "arm64be" if _is_be() else "arm64le"
    if ph.id == PLFM_ARM and ph.flag & PR_USE32:
        return "armbe" if _is_be() else "armle"
    if ph.id == PLFM_MIPS and ph.flag & PR_USE64:
        return "mips64be" if _is_be() else "mips64le"
    if ph.id == PLFM_MIPS and ph.flag & PR_USE32:
        return "mipsbe" if _is_be() else "mipsle"
    return ""


# Unicorn setup table: arch-key → (pc_const, UC_ARCH_*, UC_MODE_*)
_UC_SETUP: dict = {}   # populated lazily to avoid import-time failure

def _ensure_uc_setup():
    global _UC_SETUP
    if _UC_SETUP:
        return
    from unicorn import (
        UC_ARCH_X86, UC_ARCH_ARM, UC_ARCH_ARM64, UC_ARCH_MIPS,
        UC_MODE_32, UC_MODE_64, UC_MODE_ARM,
        UC_MODE_BIG_ENDIAN, UC_MODE_LITTLE_ENDIAN,
        UC_MODE_MIPS32, UC_MODE_MIPS64,
    )
    from unicorn.x86_const  import UC_X86_REG_RIP, UC_X86_REG_EIP
    from unicorn.arm_const   import UC_ARM_REG_PC
    from unicorn.arm64_const import UC_ARM64_REG_PC
    from unicorn.mips_const  import UC_MIPS_REG_PC

    _UC_SETUP = {
        "x64":      (UC_X86_REG_RIP,  UC_ARCH_X86,   UC_MODE_64),
        "x86":      (UC_X86_REG_EIP,  UC_ARCH_X86,   UC_MODE_32),
        "arm64be":  (UC_ARM64_REG_PC, UC_ARCH_ARM64, UC_MODE_ARM | UC_MODE_BIG_ENDIAN),
        "arm64le":  (UC_ARM64_REG_PC, UC_ARCH_ARM64, UC_MODE_ARM | UC_MODE_LITTLE_ENDIAN),
        "armbe":    (UC_ARM_REG_PC,   UC_ARCH_ARM,   UC_MODE_ARM | UC_MODE_BIG_ENDIAN),
        "armle":    (UC_ARM_REG_PC,   UC_ARCH_ARM,   UC_MODE_ARM | UC_MODE_LITTLE_ENDIAN),
        "mips64be": (UC_MIPS_REG_PC,  UC_ARCH_MIPS,  UC_MODE_MIPS64 | UC_MODE_BIG_ENDIAN),
        "mips64le": (UC_MIPS_REG_PC,  UC_ARCH_MIPS,  UC_MODE_MIPS64 | UC_MODE_LITTLE_ENDIAN),
        "mipsbe":   (UC_MIPS_REG_PC,  UC_ARCH_MIPS,  UC_MODE_MIPS32 | UC_MODE_BIG_ENDIAN),
        "mipsle":   (UC_MIPS_REG_PC,  UC_ARCH_MIPS,  UC_MODE_MIPS32 | UC_MODE_LITTLE_ENDIAN),
    }


def _build_register_map(arch: str) -> list[tuple[str, int]]:
    """Return [(name, UC_REG_CONST), ...] for the given arch."""
    from unicorn.x86_const import (
        UC_X86_REG_RAX, UC_X86_REG_RBX, UC_X86_REG_RCX, UC_X86_REG_RDX,
        UC_X86_REG_RSI, UC_X86_REG_RDI, UC_X86_REG_RBP, UC_X86_REG_RSP,
        UC_X86_REG_R8,  UC_X86_REG_R9,  UC_X86_REG_R10, UC_X86_REG_R11,
        UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15,
        UC_X86_REG_RIP, UC_X86_REG_EFLAGS,
        UC_X86_REG_EAX, UC_X86_REG_EBX, UC_X86_REG_ECX, UC_X86_REG_EDX,
        UC_X86_REG_ESI, UC_X86_REG_EDI, UC_X86_REG_EBP, UC_X86_REG_ESP,
        UC_X86_REG_EIP,
    )
    from unicorn.arm_const import (
        UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2,  UC_ARM_REG_R3,
        UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6,  UC_ARM_REG_R7,
        UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11,
        UC_ARM_REG_R12, UC_ARM_REG_PC, UC_ARM_REG_SP, UC_ARM_REG_LR,
        UC_ARM_REG_CPSR,
    )
    from unicorn.arm64_const import (
        UC_ARM64_REG_X0,  UC_ARM64_REG_X1,  UC_ARM64_REG_X2,  UC_ARM64_REG_X3,
        UC_ARM64_REG_X4,  UC_ARM64_REG_X5,  UC_ARM64_REG_X6,  UC_ARM64_REG_X7,
        UC_ARM64_REG_X8,  UC_ARM64_REG_X9,  UC_ARM64_REG_X10, UC_ARM64_REG_X11,
        UC_ARM64_REG_X12, UC_ARM64_REG_X13, UC_ARM64_REG_X14, UC_ARM64_REG_X15,
        UC_ARM64_REG_X16, UC_ARM64_REG_X17, UC_ARM64_REG_X18, UC_ARM64_REG_X19,
        UC_ARM64_REG_X20, UC_ARM64_REG_X21, UC_ARM64_REG_X22, UC_ARM64_REG_X23,
        UC_ARM64_REG_X24, UC_ARM64_REG_X25, UC_ARM64_REG_X26, UC_ARM64_REG_X27,
        UC_ARM64_REG_X28, UC_ARM64_REG_PC,  UC_ARM64_REG_SP,  UC_ARM64_REG_FP,
        UC_ARM64_REG_LR,  UC_ARM64_REG_NZCV,
    )
    from unicorn.mips_const import (
        UC_MIPS_REG_0,  UC_MIPS_REG_1,  UC_MIPS_REG_2,  UC_MIPS_REG_3,
        UC_MIPS_REG_4,  UC_MIPS_REG_5,  UC_MIPS_REG_6,  UC_MIPS_REG_7,
        UC_MIPS_REG_8,  UC_MIPS_REG_9,  UC_MIPS_REG_10, UC_MIPS_REG_11,
        UC_MIPS_REG_12, UC_MIPS_REG_13, UC_MIPS_REG_14, UC_MIPS_REG_15,
        UC_MIPS_REG_16, UC_MIPS_REG_17, UC_MIPS_REG_18, UC_MIPS_REG_19,
        UC_MIPS_REG_20, UC_MIPS_REG_21, UC_MIPS_REG_22, UC_MIPS_REG_23,
        UC_MIPS_REG_24, UC_MIPS_REG_25, UC_MIPS_REG_26, UC_MIPS_REG_27,
        UC_MIPS_REG_28, UC_MIPS_REG_29, UC_MIPS_REG_30, UC_MIPS_REG_31,
        UC_MIPS_REG_PC,
    )

    base = arch
    if arch.startswith("arm64"):
        base = "arm64"
    elif arch.startswith("arm"):
        base = "arm"
    elif arch.startswith("mips"):
        base = "mips"

    maps: dict[str, list[tuple[str, int]]] = {
        "x64": [
            ("rax", UC_X86_REG_RAX), ("rbx", UC_X86_REG_RBX), ("rcx", UC_X86_REG_RCX),
            ("rdx", UC_X86_REG_RDX), ("rsi", UC_X86_REG_RSI), ("rdi", UC_X86_REG_RDI),
            ("rbp", UC_X86_REG_RBP), ("rsp", UC_X86_REG_RSP), ("r8",  UC_X86_REG_R8),
            ("r9",  UC_X86_REG_R9),  ("r10", UC_X86_REG_R10), ("r11", UC_X86_REG_R11),
            ("r12", UC_X86_REG_R12), ("r13", UC_X86_REG_R13), ("r14", UC_X86_REG_R14),
            ("r15", UC_X86_REG_R15), ("rip", UC_X86_REG_RIP), ("rflags", UC_X86_REG_EFLAGS),
        ],
        "x86": [
            ("eax", UC_X86_REG_EAX), ("ebx", UC_X86_REG_EBX), ("ecx", UC_X86_REG_ECX),
            ("edx", UC_X86_REG_EDX), ("esi", UC_X86_REG_ESI), ("edi", UC_X86_REG_EDI),
            ("ebp", UC_X86_REG_EBP), ("esp", UC_X86_REG_ESP), ("eip", UC_X86_REG_EIP),
            ("eflags", UC_X86_REG_EFLAGS),
        ],
        "arm": [
            ("R0", UC_ARM_REG_R0), ("R1", UC_ARM_REG_R1), ("R2", UC_ARM_REG_R2),
            ("R3", UC_ARM_REG_R3), ("R4", UC_ARM_REG_R4), ("R5", UC_ARM_REG_R5),
            ("R6", UC_ARM_REG_R6), ("R7", UC_ARM_REG_R7), ("R8", UC_ARM_REG_R8),
            ("R9", UC_ARM_REG_R9), ("R10", UC_ARM_REG_R10), ("R11", UC_ARM_REG_R11),
            ("R12", UC_ARM_REG_R12), ("PC", UC_ARM_REG_PC), ("SP", UC_ARM_REG_SP),
            ("LR", UC_ARM_REG_LR), ("CPSR", UC_ARM_REG_CPSR),
        ],
        "arm64": [
            ("X0",  UC_ARM64_REG_X0),  ("X1",  UC_ARM64_REG_X1),  ("X2",  UC_ARM64_REG_X2),
            ("X3",  UC_ARM64_REG_X3),  ("X4",  UC_ARM64_REG_X4),  ("X5",  UC_ARM64_REG_X5),
            ("X6",  UC_ARM64_REG_X6),  ("X7",  UC_ARM64_REG_X7),  ("X8",  UC_ARM64_REG_X8),
            ("X9",  UC_ARM64_REG_X9),  ("X10", UC_ARM64_REG_X10), ("X11", UC_ARM64_REG_X11),
            ("X12", UC_ARM64_REG_X12), ("X13", UC_ARM64_REG_X13), ("X14", UC_ARM64_REG_X14),
            ("X15", UC_ARM64_REG_X15), ("X16", UC_ARM64_REG_X16), ("X17", UC_ARM64_REG_X17),
            ("X18", UC_ARM64_REG_X18), ("X19", UC_ARM64_REG_X19), ("X20", UC_ARM64_REG_X20),
            ("X21", UC_ARM64_REG_X21), ("X22", UC_ARM64_REG_X22), ("X23", UC_ARM64_REG_X23),
            ("X24", UC_ARM64_REG_X24), ("X25", UC_ARM64_REG_X25), ("X26", UC_ARM64_REG_X26),
            ("X27", UC_ARM64_REG_X27), ("X28", UC_ARM64_REG_X28), ("PC",  UC_ARM64_REG_PC),
            ("SP",  UC_ARM64_REG_SP),  ("FP",  UC_ARM64_REG_FP),  ("LR",  UC_ARM64_REG_LR),
            ("NZCV", UC_ARM64_REG_NZCV),
        ],
        "mips": [
            ("zero", UC_MIPS_REG_0),  ("at", UC_MIPS_REG_1), ("v0", UC_MIPS_REG_2),
            ("v1",   UC_MIPS_REG_3),  ("a0", UC_MIPS_REG_4), ("a1", UC_MIPS_REG_5),
            ("a2",   UC_MIPS_REG_6),  ("a3", UC_MIPS_REG_7), ("t0", UC_MIPS_REG_8),
            ("t1",   UC_MIPS_REG_9),  ("t2", UC_MIPS_REG_10),("t3", UC_MIPS_REG_11),
            ("t4",   UC_MIPS_REG_12), ("t5", UC_MIPS_REG_13),("t6", UC_MIPS_REG_14),
            ("t7",   UC_MIPS_REG_15), ("s0", UC_MIPS_REG_16),("s1", UC_MIPS_REG_17),
            ("s2",   UC_MIPS_REG_18), ("s3", UC_MIPS_REG_19),("s4", UC_MIPS_REG_20),
            ("s5",   UC_MIPS_REG_21), ("s6", UC_MIPS_REG_22),("s7", UC_MIPS_REG_23),
            ("t8",   UC_MIPS_REG_24), ("t9", UC_MIPS_REG_25),("k0", UC_MIPS_REG_26),
            ("k1",   UC_MIPS_REG_27), ("gp", UC_MIPS_REG_28),("sp", UC_MIPS_REG_29),
            ("fp",   UC_MIPS_REG_30), ("ra", UC_MIPS_REG_31),("pc", UC_MIPS_REG_PC),
        ],
    }
    return maps[base]


def _is_thumb(ea: int) -> bool:
    """True when ea is a Thumb instruction (ARM 32-bit only)."""
    try:
        import idaapi
        if idaapi.ph.id != idaapi.PLFM_ARM or idaapi.ph.flag & idaapi.PR_USE64:
            return False
        t = idaapi.get_sreg(ea, "T")
        return t not in (idaapi.BADSEL, 0)
    except Exception:
        return False


def _map_and_copy_segments(mu) -> None:
    """Map every IDA segment into Unicorn and copy initialised bytes."""
    import idaapi
    import idc
    from idautils import Segments

    last_end = 0
    for seg_ea in Segments():
        seg_start = idc.get_segm_start(seg_ea)
        seg_end   = idc.get_segm_end(seg_ea)
        if seg_start >= seg_end:
            continue

        aligned_start = _align_down(seg_start)
        aligned_end   = _align_up(seg_end)

        # Try to map; skip if overlapping with already-mapped region
        try:
            mu.mem_map(aligned_start, aligned_end - aligned_start)
        except Exception:
            # Unicorn raises if region overlaps — just continue
            pass

        # Copy initialised bytes
        data = idaapi.get_bytes(seg_start, seg_end - seg_start)
        if data:
            try:
                mu.mem_write(seg_start, bytes(data))
            except Exception:
                pass

    # Also map a default stack at a high address if not already there
    _ensure_stack(mu)


_DEFAULT_STACK_BASE = 0x7FFFFFFF0000
_DEFAULT_STACK_SIZE = 0x10000

def _ensure_stack(mu) -> int:
    """Map a scratch stack if nothing is mapped at _DEFAULT_STACK_BASE yet."""
    for start, end, _ in mu.mem_regions():
        if start <= _DEFAULT_STACK_BASE < end:
            return _DEFAULT_STACK_BASE + _DEFAULT_STACK_SIZE // 2
    try:
        mu.mem_map(_DEFAULT_STACK_BASE, _DEFAULT_STACK_SIZE)
        mu.mem_write(_DEFAULT_STACK_BASE, b"\x00" * _DEFAULT_STACK_SIZE)
    except Exception:
        pass
    return _DEFAULT_STACK_BASE + _DEFAULT_STACK_SIZE // 2


def _apply_registers(mu, reg_map: list[tuple[str, int]], overrides: dict[str, str]) -> None:
    """Write user-supplied register values into Unicorn (case-insensitive keys)."""
    lower = {k.lower(): v for k, v in overrides.items()}
    for name, const in reg_map:
        val_str = lower.get(name.lower())
        if val_str is not None:
            mu.reg_write(const, int(val_str, 0))


def _read_registers(mu, reg_map: list[tuple[str, int]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, const in reg_map:
        try:
            out[name] = hex(mu.reg_read(const))
        except Exception:
            out[name] = "0x0"
    return out


def _read_memory_regions(mu, regions: list[dict]) -> list[EmuMemRead]:
    results: list[EmuMemRead] = []
    for r in regions:
        try:
            addr = parse_address(r.get("address", "0"))
            size = int(r.get("size", 0))
            if size <= 0:
                continue
            data = bytes(mu.mem_read(addr, size))
            results.append(EmuMemRead(
                address=hex(addr),
                size=size,
                data=data.hex(),
            ))
        except Exception as e:
            results.append(EmuMemRead(address=r.get("address", "?"), size=0, data=f"error: {e}"))
    return results


def _stop_hook_factory(stop_ea: int, insn_counter: list[int]):
    """Return a code hook that stops emulation at stop_ea."""
    def _hook(uc, address, size, user_data):
        insn_counter[0] += 1
        if address == stop_ea:
            uc.emu_stop()
    return _hook


def _run_emulation(
    start_ea: int,
    end_ea: int,
    arch: str,
    regs_override: dict[str, str],
    max_insns: int,
    timeout_us: int,
    memory_reads: list[dict],
) -> EmuResult:
    """Core emulation routine; must be called from the IDA main thread."""
    from unicorn import Uc, UcError, UC_HOOK_CODE, UC_HOOK_MEM_UNMAPPED

    _ensure_uc_setup()
    if arch not in _UC_SETUP:
        raise IDAError(f"Unsupported architecture: {arch!r}")

    uc_reg_pc, uc_arch, uc_mode = _UC_SETUP[arch]
    reg_map = _build_register_map(arch)

    mu = Uc(uc_arch, uc_mode)

    # Map IDA segments into Unicorn
    _map_and_copy_segments(mu)

    # Set default stack pointer to the middle of our scratch stack
    sp_mid = _DEFAULT_STACK_BASE + _DEFAULT_STACK_SIZE // 2
    sp_name = {"x64": "rsp", "x86": "esp", "arm": "SP", "arm64": "SP", "mips": "sp"}
    base_arch = arch
    for prefix in ("arm64", "arm", "mips"):
        if arch.startswith(prefix):
            base_arch = prefix
            break
    sp_key = sp_name.get(base_arch, "")
    if sp_key and sp_key.lower() not in {k.lower() for k in regs_override}:
        for name, const in reg_map:
            if name.lower() == sp_key.lower():
                mu.reg_write(const, sp_mid)
                break

    # Apply user overrides
    _apply_registers(mu, reg_map, regs_override)

    # Set PC
    mu.reg_write(uc_reg_pc, start_ea)

    # Instruction counter and hooks
    insn_counter: list[int] = [0]

    def _code_hook(uc, address, size, user_data):
        insn_counter[0] += 1

    mu.hook_add(UC_HOOK_CODE, _code_hook)

    stop_reason = "completed"
    uc_error: Optional[str] = None

    def _mem_err_hook(uc, access, address, size, value, user_data):
        nonlocal stop_reason
        access_type = {1: "read", 2: "write", 4: "fetch"}.get(access, "access")
        stop_reason = f"unmapped_memory_{access_type}:0x{address:x}"
        uc.emu_stop()
        return False

    mu.hook_add(UC_HOOK_MEM_UNMAPPED, _mem_err_hook)

    try:
        start_addr = start_ea | 1 if _is_thumb(start_ea) else start_ea
        mu.emu_start(start_addr, end_ea, timeout=timeout_us, count=max_insns)
    except UcError as e:
        stop_reason = f"unicorn_error:{e}"
        uc_error = str(e)

    # If the emulator ran out of instruction budget
    if stop_reason == "completed" and max_insns > 0 and insn_counter[0] >= max_insns:
        stop_reason = "max_instructions_reached"

    final_regs = _read_registers(mu, reg_map)
    mem_results = _read_memory_regions(mu, memory_reads)

    result = EmuResult(
        start=hex(start_ea),
        end=hex(end_ea),
        arch=arch,
        stop_reason=stop_reason,
        instructions_executed=insn_counter[0],
        registers=final_regs,
        memory_reads=mem_results,
    )
    if uc_error:
        result["error"] = uc_error
    return result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@ext("emu")
@tool
@idasync
def emulate_func(
    addr: Annotated[
        str,
        "Function address or name to emulate from (hex or symbol, e.g. '0x401000' or 'sub_401000').",
    ],
    registers: Annotated[
        Optional[dict[str, str]],
        (
            "Initial register values as {name: hex_value}, e.g. "
            '{"rdi": "0x1", "rsi": "0x4000"}. '
            "Omit to use zeroed registers. "
            "Stack pointer is set to a scratch region automatically unless overridden."
        ),
    ] = None,
    max_insns: Annotated[
        int,
        "Maximum number of instructions to execute (0 = unlimited, default 10000).",
    ] = 10000,
    timeout_us: Annotated[
        int,
        "Emulation timeout in microseconds (0 = no timeout, default 5000000 = 5 s).",
    ] = 5_000_000,
    memory_reads: Annotated[
        Optional[list[dict]],
        (
            "Memory regions to read back after emulation, each as "
            '{"address": "0x...", "size": N}. Returned as hex strings in memory_reads.'
        ),
    ] = None,
) -> EmuResult:
    """Emulate a function using Unicorn Engine and return the final CPU state.

    Maps all IDA segments into Unicorn before starting so that code and
    global data are already available.  Emulation stops when a RET/BX LR/etc.
    is executed (the end address is set to 0xFFFFFFFFFFFFFFFF so Unicorn
    catches the implied return), the instruction limit is hit, an unmapped
    memory access occurs, or the timeout expires.

    Typical use — emulate a decryption routine with known inputs:
        emulate_func("sub_401000", registers={"rdi": "0x402000", "rsi": "0x10"})

    After the call, read the result registers or use memory_reads to inspect
    output buffers.

    Note: Unicorn must be installed in IDA's Python environment (pip install unicorn).
    """
    try:
        from unicorn import Uc  # noqa: F401 — presence check
    except ImportError:
        raise IDAError(
            "unicorn is not installed in IDA's Python environment. "
            "Run: pip install unicorn"
        )

    import idaapi
    ea = parse_address(addr)
    # Find function end: try func_t, otherwise use a heuristic
    func = idaapi.get_func(ea)
    if func is not None:
        func_end = func.end_ea
    else:
        func_end = 0xFFFFFFFFFFFFFFFF

    arch = _get_arch()
    if not arch:
        raise IDAError("Unsupported or unrecognised processor architecture.")

    return _run_emulation(
        start_ea=ea,
        end_ea=func_end,
        arch=arch,
        regs_override=registers or {},
        max_insns=max_insns,
        timeout_us=timeout_us,
        memory_reads=memory_reads or [],
    )


@ext("emu")
@tool
@idasync
def emulate_range(
    start: Annotated[
        str,
        "Start address of the instruction range (hex or symbol, e.g. '0x401000').",
    ],
    end: Annotated[
        str,
        "Exclusive end address — emulation stops when the PC reaches this address.",
    ],
    registers: Annotated[
        Optional[dict[str, str]],
        (
            "Initial register values as {name: hex_value}, e.g. "
            '{"rdi": "0x1", "rsi": "0x4000"}. '
            "Omit to use zeroed registers. "
            "Stack pointer is set to a scratch region automatically unless overridden."
        ),
    ] = None,
    max_insns: Annotated[
        int,
        "Maximum instructions to execute (0 = unlimited, default 10000).",
    ] = 10000,
    timeout_us: Annotated[
        int,
        "Emulation timeout in microseconds (0 = no timeout, default 5000000 = 5 s).",
    ] = 5_000_000,
    memory_reads: Annotated[
        Optional[list[dict]],
        (
            "Memory regions to read back after emulation, each as "
            '{"address": "0x...", "size": N}. Returned as hex strings in memory_reads.'
        ),
    ] = None,
) -> EmuResult:
    """Emulate a specific address range using Unicorn Engine.

    Emulation begins at `start` and stops when the PC reaches `end` (or the
    instruction / timeout limit is hit).  Useful for emulating a loop, a
    decryption stub, or any self-contained code region without emulating an
    entire function.

    Example — emulate a short XOR loop between two labels:
        emulate_range("0x401010", "0x401030",
                      registers={"rcx": "0x10", "rsi": "0x402000"},
                      memory_reads=[{"address": "0x402000", "size": 16}])

    Note: Unicorn must be installed in IDA's Python environment (pip install unicorn).
    """
    try:
        from unicorn import Uc  # noqa: F401 — presence check
    except ImportError:
        raise IDAError(
            "unicorn is not installed in IDA's Python environment. "
            "Run: pip install unicorn"
        )

    arch = _get_arch()
    if not arch:
        raise IDAError("Unsupported or unrecognised processor architecture.")

    start_ea = parse_address(start)
    end_ea   = parse_address(end)
    if end_ea <= start_ea:
        raise IDAError(f"end ({hex(end_ea)}) must be greater than start ({hex(start_ea)})")

    return _run_emulation(
        start_ea=start_ea,
        end_ea=end_ea,
        arch=arch,
        regs_override=registers or {},
        max_insns=max_insns,
        timeout_us=timeout_us,
        memory_reads=memory_reads or [],
    )
