"""Private standard-library runner inside the FT data processing sandbox."""
import resource
import runpy
import sys

resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
sys.argv = ['/work/process_data.py']
runpy.run_path('/work/process_data.py', run_name='__main__')
