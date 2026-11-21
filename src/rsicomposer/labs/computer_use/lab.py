"""Run WebArena-Infinity tasks against a fresh local application per trial."""

from functools import partial
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.request import urlopen

from rsicomposer.labs._task_evaluation import TaskEvaluationLab


def load_module(path):
    spec = spec_from_file_location('_webarena_infinity', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ComputerUseLab(TaskEvaluationLab):
    domain = 'computer_use'
    adapters = {'webarena-infinity': ('success_rate', 'webarena-infinity-state-verifier/v1',
                                     {'checkout', 'revision'})}

    def measure(self, benchmark, task, instruction):
        from playwright.sync_api import sync_playwright

        entry = self.entries[benchmark]
        root = Path(entry['checkout']).resolve(strict=True)
        if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != entry['revision']:
            raise ValueError('WebArena-Infinity checkout revision does not match its binding')
        subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--'], check=True)
        reference = task['reference']
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', reference['app']):
            raise ValueError('Expected a WebArena-Infinity app name')
        app = root / 'apps' / reference['app']
        path = (app / reference['verify']).resolve(strict=True)
        if not path.is_relative_to(app / 'real-tasks'):
            raise ValueError('Expected an app real-task verifier')
        server_module = load_module(app / 'server.py')
        # The app serves only public UI assets, so the verifier and task suite
        # are never reachable through the browser's origin.
        with TemporaryDirectory(prefix='rsisys-webarena-') as work:
            shutil.copyfile(app / 'index.html', Path(work) / 'index.html')
            for directory in ('css', 'js'):
                shutil.copytree(app / directory, Path(work) / directory)
            handler = partial(server_module.AppHandler, directory=work)
            with server_module.ThreadedHTTPServer(('127.0.0.1', 0), handler) as server:
                origin = f'http://127.0.0.1:{server.server_port}'
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with sync_playwright() as playwright:
                        browser = playwright.chromium.launch()
                        session = browser.new_context(service_workers='block')
                        session.route('**/*', lambda route: route.continue_()
                                      if route.request.url.startswith(origin + '/') else route.abort())
                        page = session.new_page()
                        page.goto(origin, wait_until='domcontentloaded')
                        page.wait_for_function('() => fetch("/api/state").then(r => r.ok)', timeout=10000)
                        trace = self.runner(instruction=instruction, task=dict(task['input']),
                                            context=self.context, page=page)
                        with urlopen(origin + '/api/state', timeout=10) as response:
                            if response.status != 200:
                                raise RuntimeError('WebArena-Infinity application state is unavailable')
                        passed, message = load_module(path).verify(origin)
                        if type(passed) is not bool:
                            raise ValueError('WebArena-Infinity verifier did not return a boolean')
                        session.close()
                        browser.close()
                finally:
                    server.shutdown()
                    thread.join()
        return {'success_rate': int(passed)}, {'actions': trace, 'verifier_message': message}
