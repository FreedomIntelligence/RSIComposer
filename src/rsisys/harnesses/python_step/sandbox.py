"""Private, stdlib-only runner; executed inside a fresh Linux namespace."""

import contextlib
import io
import json
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
request = json.load(sys.stdin)
namespace = {}
covered = set()


def trace(frame, event, arg):
    if frame.f_code.co_filename == "/harness.py" and event == "line":
        covered.add(frame.f_lineno)
    return trace


with contextlib.redirect_stdout(io.StringIO()):
    sys.settrace(trace)
    exec(compile(open("/harness.py").read(), "/harness.py", "exec"), namespace)
    action = namespace["step"](request["problem"], request["history"])
    sys.settrace(None)
result = json.dumps(
    {"action": action, "covered_lines": sorted(covered)}, allow_nan=False
)
if len(result) > 128000:
    raise ValueError("Oversize harness output")
print(result)
