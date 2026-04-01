"""
Sandboxed code execution for HumanEval benchmark.
Executes generated code in an isolated subprocess with timeout.
"""
import subprocess
import tempfile
import os
import asyncio
from typing import Optional


async def execute_code_safely(
    code: str,
    test_code: str,
    timeout: int = 10,
) -> tuple[bool, Optional[str]]:
    """
    Execute generated code against test cases in a sandboxed subprocess.

    Args:
        code: The generated Python code to test
        test_code: The test assertions to run
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success: bool, error_message: str | None)
    """
    full_code = f"{code}\n\n{test_code}"

    # Create temp file for execution
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(full_code)

        # Run in subprocess with timeout
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["python", path],
                    capture_output=True,
                    timeout=timeout,
                )
            )
            if result.returncode == 0:
                return True, None
            else:
                error_msg = result.stderr.decode("utf-8", errors="replace")[-500:]
                return False, error_msg
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {timeout}s"
        except Exception as e:
            return False, str(e)
    finally:
        # Clean up temp file
        try:
            os.unlink(path)
        except OSError:
            pass


def extract_function_code(response: str, entry_point: str) -> Optional[str]:
    """
    Extract the function code from model response.

    Args:
        response: Full model response text
        entry_point: Expected function name

    Returns:
        Extracted function code or None
    """
    # Try to find a complete function definition
    lines = response.split("\n")
    code_lines = []
    in_function = False
    indent_level = None

    for line in lines:
        # Start of function
        if f"def {entry_point}" in line:
            in_function = True
            code_lines.append(line)
            indent_level = len(line) - len(line.lstrip())
            continue

        if in_function:
            # Check if we've exited the function
            stripped = line.strip()
            if stripped and not line.startswith(" " * (indent_level + 1)) and not line.startswith("\t"):
                # Non-indented, non-empty line = end of function
                if not stripped.startswith("def ") and not stripped.startswith("@"):
                    break

            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    # Fallback: return entire response wrapped in a function attempt
    if "def " not in response:
        return f"def {entry_point}():\n    " + "\n    ".join(response.split("\n"))

    return response


async def run_humaneval_test(
    prompt: str,
    response: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> dict:
    """
    Run a single HumanEval test case.

    Args:
        prompt: The original prompt (function signature + docstring)
        response: Model's generated code
        test_code: The test assertions
        entry_point: Expected function name
        timeout: Execution timeout

    Returns:
        Dict with "passed" (bool) and "error" (str | None)
    """
    # Extract the function code from response
    code = extract_function_code(response, entry_point)

    # Combine prompt (signature) + response (implementation)
    # The prompt already contains the function signature
    full_code = response

    # If response doesn't include the signature, prepend it
    if "def " not in response or entry_point not in response:
        # Extract just the signature from prompt
        signature_lines = []
        for line in prompt.split("\n"):
            signature_lines.append(line)
            if line.strip().endswith(":"):
                break
        signature = "\n".join(signature_lines)

        # Combine signature with response body
        full_code = signature + "\n" + "\n".join(
            "    " + line for line in response.split("\n") if line.strip()
        )

    success, error = await execute_code_safely(full_code, test_code, timeout)

    return {
        "passed": success,
        "error": error,
        "code": full_code[:500] if not success else None,  # Include code on failure
    }