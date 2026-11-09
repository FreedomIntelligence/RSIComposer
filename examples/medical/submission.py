"""Host packaging/preflight. No model loading, execution, scoring or quota use."""
import ast
from pathlib import Path
import shutil
import tempfile

from rsisys.runtime.bundles import workspace_path
from rsisys.runtime.files import atomic_write, json_bytes
from .protocol_evaluation import BUNDLE_SCHEMA, bundle_configuration, bundle_manifest, digest


def preflight(root, bases):
    manifest = bundle_manifest(root)
    config = bundle_configuration(root, bases)
    ast.parse((Path(root) / config['entrypoint']).read_text(encoding='utf-8'))
    for field in ('adapter_path', 'checkpoint_path'):
        if field not in config:
            continue
        directory = Path(root) / config[field]
        if not any(directory.glob('*.safetensors')) and not any(directory.glob('*.bin')):
            raise ValueError(field + ' contains no weight files')
        if field == 'adapter_path' and not (directory / 'adapter_config.json').is_file():
            raise ValueError('LoRA adapter requires adapter_config.json')
    return {'status': 'structurally_valid', 'configuration': config,
            'bundle_sha256': digest(manifest), 'files': len(manifest),
            'model_loaded': False, 'evaluation_submitted': False,
            'instruction': 'Structural validation only. Test loading/inference before formal evaluation.'}


def package_submission(workspace, bases, *, destination_path, entrypoint_path, base_model_path,
                       adapter_path=None, checkpoint_path=None):
    if adapter_path is not None and checkpoint_path is not None:
        raise ValueError('Choose checkpoint OR LoRA')
    entry = workspace_path(workspace, entrypoint_path)
    destination = workspace_path(workspace, destination_path)
    if not entry.is_file() or entry.suffix != '.py' or destination.exists():
        raise ValueError('Use an existing Python entrypoint and a fresh destination')
    if base_model_path not in bases:
        raise ValueError('Choose a registered base model')
    sources = {}
    for field, value in (('adapter_path', adapter_path), ('checkpoint_path', checkpoint_path)):
        if value is not None:
            source = workspace_path(workspace, value)
            if not source.is_dir() or destination.is_relative_to(source):
                raise ValueError('Weights must be an existing directory separate from the destination')
            sources[field] = source
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.submission-', dir=destination.parent))
    try:
        # No symlinks/hardlinks: generated artifacts and bundles evolve independently.
        shutil.copyfile(entry, stage / 'infer.py')
        config = {'schema': BUNDLE_SCHEMA, 'entrypoint': 'infer.py', 'base_model_path': base_model_path}
        for field, source in sources.items():
            before = bundle_manifest(source)
            shutil.copytree(source, stage / 'weights')
            if bundle_manifest(source) != before or bundle_manifest(stage / 'weights') != before:
                raise ValueError('Weights changed during packaging')
            config[field] = 'weights'
        atomic_write(stage / 'submission.json', json_bytes(config))
        result = preflight(stage, bases)
        if destination.exists():
            raise ValueError('Submission destination already exists')
        stage.rename(destination)
        return {**result, 'submission_path': destination_path,
                'copied_files': ['infer.py', 'weights (if supplied)'],
                'instruction': 'Copy any additional Python modules/config into the bundle and rerun '
                               'medical_preflight. External workspace dependencies are not mounted.'}
    finally:
        if stage.exists():
            shutil.rmtree(stage)
