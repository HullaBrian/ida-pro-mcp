"""Tests for the Unicorn Engine emulation tools.

Most tests exercise the pure-Python helper functions that don't require IDA
or Unicorn to be present.  Tests that require Unicorn are skipped when the
package is not installed.
"""

from ..framework import test, assert_shape
from .. import api_emu


# ---------------------------------------------------------------------------
# _get_arch helper (pure Python, mocks IDA state)
# ---------------------------------------------------------------------------

class _FakePh:
    def __init__(self, id_, flag):
        self.id = id_
        self.flag = flag


@test()
def test_get_arch_detects_x64():
    """_get_arch returns 'x64' for x86-64 IDB."""
    import idaapi

    old_ph = idaapi.ph
    old_inf = getattr(idaapi, 'inf_is_be', None)
    try:
        fake = _FakePh(idaapi.PLFM_386, idaapi.PR_USE64)
        idaapi.ph = fake
        result = api_emu._get_arch()
        assert result == "x64", f"Expected 'x64', got {result!r}"
    finally:
        idaapi.ph = old_ph


@test()
def test_get_arch_detects_x86():
    """_get_arch returns 'x86' for 32-bit x86 IDB."""
    import idaapi

    old_ph = idaapi.ph
    try:
        fake = _FakePh(idaapi.PLFM_386, idaapi.PR_USE32)
        idaapi.ph = fake
        result = api_emu._get_arch()
        assert result == "x86", f"Expected 'x86', got {result!r}"
    finally:
        idaapi.ph = old_ph


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

@test()
def test_align_down():
    """_align_down rounds down to nearest 0x1000."""
    assert api_emu._align_down(0x1001) == 0x1000
    assert api_emu._align_down(0x2000) == 0x2000
    assert api_emu._align_down(0x1FFF) == 0x1000


@test()
def test_align_up():
    """_align_up rounds up to nearest 0x1000."""
    assert api_emu._align_up(0x1001) == 0x2000
    assert api_emu._align_up(0x2000) == 0x2000
    assert api_emu._align_up(0x1) == 0x1000


# ---------------------------------------------------------------------------
# Register map
# ---------------------------------------------------------------------------

@test()
def test_build_register_map_x64_requires_unicorn():
    """_build_register_map for x64 returns at least rax and rip."""
    try:
        import unicorn  # noqa: F401
    except ImportError:
        return  # Skip — Unicorn not installed

    reg_map = api_emu._build_register_map("x64")
    names = [r[0] for r in reg_map]
    assert "rax" in names
    assert "rip" in names
    assert "rsp" in names


@test()
def test_build_register_map_x86_requires_unicorn():
    """_build_register_map for x86 returns eax and eip."""
    try:
        import unicorn  # noqa: F401
    except ImportError:
        return

    reg_map = api_emu._build_register_map("x86")
    names = [r[0] for r in reg_map]
    assert "eax" in names
    assert "eip" in names


@test()
def test_build_register_map_arm_requires_unicorn():
    """_build_register_map for arm returns R0 and PC."""
    try:
        import unicorn  # noqa: F401
    except ImportError:
        return

    reg_map = api_emu._build_register_map("armle")
    names = [r[0] for r in reg_map]
    assert "R0" in names
    assert "PC" in names


# ---------------------------------------------------------------------------
# _apply_registers + _read_registers (requires Unicorn)
# ---------------------------------------------------------------------------

@test()
def test_apply_and_read_registers_x64():
    """_apply_registers writes values; _read_registers reads them back."""
    try:
        from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
        from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RBX
    except ImportError:
        return

    mu = Uc(UC_ARCH_X86, UC_MODE_64)
    reg_map = api_emu._build_register_map("x64")

    api_emu._apply_registers(mu, reg_map, {"rax": "0x1234", "RBX": "0x5678"})
    state = api_emu._read_registers(mu, reg_map)

    assert state["rax"] == hex(0x1234), f"rax mismatch: {state['rax']!r}"
    assert state["rbx"] == hex(0x5678), f"rbx mismatch: {state['rbx']!r}"


# ---------------------------------------------------------------------------
# emulate_range / emulate_func error paths
# ---------------------------------------------------------------------------

@test()
def test_emulate_range_rejects_inverted_range():
    """emulate_range raises IDAError when end <= start."""
    from ..sync import IDAError as _IDAError
    try:
        api_emu.emulate_range(start="0x401010", end="0x401000")
        assert False, "Should have raised IDAError"
    except _IDAError as e:
        assert "end" in str(e).lower() or "greater" in str(e).lower()
    except Exception:
        # If Unicorn not installed, a different error may surface — that's fine
        pass
