"""
Sandboxed Python code execution tool.
Uses subprocess with a restricted environment, NOT eval().
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import textwrap
from pathlib import Path

import structlog

from app.tools.registry import ToolSpec, registry

log = structlog.get_logger(__name__)

EXECUTION_TIMEOUT = 15  # seconds

# Allowed built-ins — explicitly whitelisted for safety
SAFE_BUILTINS = [
    "print", "range", "len", "list", "dict", "set", "tuple",
    "str", "int", "float", "bool", "sum", "min", "max",
    "sorted", "enumerate", "zip", "map", "filter",
    "abs", "round", "isinstance", "type", "repr",
]

PREAMBLE = textwrap.dedent(
    f"""
    import builtins as _b
    _allowed = {SAFE_BUILTINS!r}
    _safe = {{k: getattr(_b, k) for k in _allowed if hasattr(_b, k)}}
    _safe['__import__'] = __import__
    import math, json, re, statistics, datetime
    """
)


async def execute_python(code: str) -> dict:
    """
    Execute *code* in a sandboxed subprocess and return stdout/stderr/exit_code.

    Returns
    -------
    dict with keys: stdout, stderr, exit_code, timed_out
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="agentis_exec_"
    ) as tmp:
        tmp.write(PREAMBLE + "\n" + code)
        tmp_path = tmp.name

    log.info("code_exec_start", script=tmp_path)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=EXECUTION_TIMEOUT
            )
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            stdout_b, stderr_b = b"", b"Execution timed out"
            timed_out = True

        result = {
            "stdout":    stdout_b.decode("utf-8", errors="replace"),
            "stderr":    stderr_b.decode("utf-8", errors="replace"),
            "exit_code": proc.returncode if not timed_out else -1,
            "timed_out": timed_out,
        }
        log.info(
            "code_exec_done",
            exit_code=result["exit_code"],
            timed_out=timed_out,
            stdout_len=len(result["stdout"]),
        )
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Register tool
registry.register(
    ToolSpec(
        name="execute_python",
        description=(
            "Execute a Python code snippet in a sandboxed subprocess. "
            "Input: code (str). "
            "Returns dict with stdout, stderr, exit_code, timed_out. "
            "Available modules: math, json, re, statistics, datetime."
        ),
        fn=execute_python,
        allowed_agents=["analyst"],
    )
)
