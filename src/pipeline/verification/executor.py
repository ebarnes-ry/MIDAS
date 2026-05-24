import time
import sys
from typing import Dict, Any
import ast
from io import StringIO
import traceback
import multiprocessing as mp

from .verification_types import CodeExecutionResult


ALLOWED_IMPORT_MODULES = {
    'sympy',
    'json',
    'math',
    'itertools',
    'functools',
    'operator',
    'collections'
}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Custom import function that only allows whitelisted, safe modules."""
    if name.split('.')[0] not in ALLOWED_IMPORT_MODULES:
        raise ImportError(
            f"Import of module '{name}' is not allowed. Only {ALLOWED_IMPORT_MODULES} are permitted."
        )
    return __import__(name, globals, locals, fromlist, level)


def _create_safe_namespace() -> Dict[str, Any]:
    """Creates a safe, whitelisted global namespace for code execution."""
    safe_builtins = {
        # Data types
        'str': str, 'int': int, 'float': float, 'bool': bool, 'list': list,
        'dict': dict, 'tuple': tuple, 'set': set, 'complex': complex,
        'isinstance': isinstance, 'type': type,

        # Math and Data Manipulation
        'print': print, 'len': len, 'abs': abs, 'max': max, 'min': min,
        'round': round, 'sum': sum, 'divmod': divmod, 'pow': pow,

        # Iteration
        'range': range, 'enumerate': enumerate, 'zip': zip, 'map': map,
        'filter': filter, 'sorted': sorted, 'reversed': reversed, 'all': all, 'any': any,

        # Exceptions
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        'NameError': NameError, 'IndexError': IndexError, 'KeyError': KeyError,
        'ZeroDivisionError': ZeroDivisionError,

        # The restricted import function
        '__import__': _restricted_import
    }
    return {'__builtins__': safe_builtins}


def _apply_child_resource_limits(max_memory_mb: int):
    """Applies memory limits inside the worker process only."""
    try:
        import resource
        memory_bytes = max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ImportError, ValueError):
        print("Warning: Could not set memory limits. 'resource' module not available.", file=sys.stderr)


def _execute_code_worker(code: str, max_memory_mb: int, result_queue):
    """Run verification code in an isolated child process."""
    start_time = time.time()
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    try:
        _apply_child_resource_limits(max_memory_mb)
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        exec(code, _create_safe_namespace())
        result_queue.put(
            {
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "execution_time": time.time() - start_time,
            }
        )
    except BaseException as e:
        formatted_traceback = traceback.format_exc()
        result_queue.put(
            {
                "success": False,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue() + formatted_traceback,
                "execution_time": time.time() - start_time,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "exception_traceback": formatted_traceback,
            }
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class SafeExecutor:
    """
    Executes untrusted Python code in a restricted, sandboxed environment
    with resource limits (timeout, memory).
    """
    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def _restricted_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Custom import function that only allows whitelisted, safe modules."""
        return _restricted_import(name, globals, locals, fromlist, level)

    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Creates a safe, whitelisted global namespace for code execution."""
        return _create_safe_namespace()

    def execute(self, code: str) -> CodeExecutionResult:
        """
        Executes the provided code string in the sandbox.

        Returns:
            A CodeExecutionResult object with the outcome.
        """
        start_time = time.time()

        try:
            # Preliminary check for syntax errors before execution.
            ast.parse(code)
        except Exception as e:
            formatted_traceback = traceback.format_exc()
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=formatted_traceback,
                execution_time=time.time() - start_time,
                exception_type=type(e).__name__,
                exception_message=str(e),
                exception_traceback=formatted_traceback,
            )

        # Spawn keeps verification isolated from the API worker's loaded models.
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_execute_code_worker,
            args=(code, self.max_memory_mb, result_queue),
        )
        process.start()
        process.join(self.timeout)

        if process.is_alive():
            process.terminate()
            process.join(1)
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution exceeded the time limit of {self.timeout} seconds.",
                execution_time=time.time() - start_time,
                exception_type="TimeoutError",
                exception_message=f"Execution exceeded the time limit of {self.timeout} seconds.",
            )

        if not result_queue.empty():
            payload = result_queue.get()
            payload["execution_time"] = time.time() - start_time
            return CodeExecutionResult(**payload)

        if process.exitcode not in (0, None):
            return CodeExecutionResult(
                success=False,
                stdout="",
                stderr=f"Verification worker exited unexpectedly with code {process.exitcode}.",
                execution_time=time.time() - start_time,
                exception_type="WorkerProcessError",
                exception_message=f"Verification worker exited unexpectedly with code {process.exitcode}.",
            )

        return CodeExecutionResult(
            success=False,
            stdout="",
            stderr="Verification worker finished without returning a result.",
            execution_time=time.time() - start_time,
            exception_type="WorkerProcessError",
            exception_message="Verification worker finished without returning a result.",
        )
