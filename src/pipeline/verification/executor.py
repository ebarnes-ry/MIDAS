import time
import sys
import signal
from typing import Dict, Any
import ast
from io import StringIO
import traceback
from contextlib import contextmanager

from .verification_types import CodeExecutionResult


class SafeExecutor:
    """
    Executes untrusted Python code in a restricted, sandboxed environment
    with resource limits (timeout, memory).
    """
    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    @contextmanager
    def _timeout_context(self):
        """Context manager to enforce a timeout on a block of code using SIGALRM."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded the time limit of {self.timeout} seconds.")

        try:
            original_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            yield
        finally:
            signal.alarm(0)  # Disable the alarm
            signal.signal(signal.SIGALRM, original_handler)

    def _restricted_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Custom import function that only allows whitelisted, safe modules."""
        allowed_modules = {
            'sympy', 
            'json',
            'math',
            'itertools',
            'functools',
            'operator',
            'collections'
        }
        if name.split('.')[0] not in allowed_modules:
            raise ImportError(f"Import of module '{name}' is not allowed. Only {allowed_modules} are permitted.")
        return __import__(name, globals, locals, fromlist, level)

    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Creates a safe, whitelisted global namespace for code execution."""
        
        # --- THIS IS THE NEW, EXPANDED BUILT-INS DICTIONARY ---
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
            
            # Exceptions (already fixed)
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            'NameError': NameError, 'IndexError': IndexError, 'KeyError': KeyError,
            'ZeroDivisionError': ZeroDivisionError,

            # The restricted import function
            '__import__': self._restricted_import
        }
        # --- END OF NEW BUILT-INS ---
        
        namespace = {'__builtins__': safe_builtins}
        return namespace

    def execute(self, code: str) -> CodeExecutionResult:
        """
        Executes the provided code string in the sandbox.

        Returns:
            A CodeExecutionResult object with the outcome.
        """
        start_time = time.time()
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        try:
            # Preliminary check for syntax errors before execution.
            ast.parse(code)

            # Redirect stdout and stderr to capture the output.
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Apply resource limits and execute.
            self._apply_resource_limits()
            with self._timeout_context():
                exec(code, self._create_safe_namespace())

            return CodeExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + traceback.format_exc(),
                execution_time=time.time() - start_time,
                exception_type=type(e).__name__,
                exception_message=str(e),
                exception_traceback=traceback.format_exc()
            )
        finally:
            # Restore stdout and stderr.
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _apply_resource_limits(self):
        """Applies memory limits using the resource module (POSIX-only)."""
        try:
            import resource
            memory_bytes = self.max_memory_mb * 1024 * 1024
            # Set the max virtual memory size.
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ImportError, ValueError):
            # This will fail on non-POSIX systems (e.g., Windows).
            # The pipeline will still work, but without memory protection.
            print("Warning: Could not set memory limits. 'resource' module not available.", file=sys.stderr)
