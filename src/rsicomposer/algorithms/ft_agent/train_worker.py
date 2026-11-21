"""Private native LLaMA-Factory training and numerical publication worker."""
import json
from pathlib import Path
import sys


def checkpoint_sync_callback():
    """Publish the writer's checkpoint state after a native distributed save.

    Transformers 4.52 checks checkpoint existence before all ranks finish
    writing. Peers can retain None and skip the end-of-training load barrier.
    Synchronize the native choice; do not choose another checkpoint or metric.
    """
    from transformers import TrainerCallback

    class CheckpointSync(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            import torch.distributed as dist
            if args.load_best_model_at_end and dist.is_initialized() and dist.get_world_size() > 1:
                payload = [(state.best_model_checkpoint, state.best_metric, state.best_global_step)
                           if dist.get_rank() == 0 else None]
                dist.broadcast_object_list(payload, src=0)
                state.best_model_checkpoint, state.best_metric, state.best_global_step = payload[0]
            return control

    return CheckpointSync()


def main(request):
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    directory = Path(request['directory'])
    recipe = dict(request['recipe'])
    mode = request['mode']
    if mode in ('micro', 'train'):
        from llamafactory.train.tuner import run_exp
        if mode == 'micro':
            # A separate one-step execution check; never published or scored.
            recipe.update(max_steps=1, output_dir=str(directory / 'micro-output'),
                tokenized_path=str(directory / 'micro-tokens'), save_strategy='no',
                eval_strategy='no', load_best_model_at_end=False, report_to='none',
                plot_loss=False)
        run_exp(args=recipe, callbacks=[checkpoint_sync_callback()])
        print('RSISYS_NATIVE_TRAINING_RETURNED ' + mode, flush=True)
        return
    if mode != 'merge':
        raise ValueError('Unknown LLaMA-Factory worker mode')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rsicomposer.api import merged_change_count
    output = Path(recipe['output_dir'])
    states = sorted(output.rglob('trainer_state.json'), key=lambda p: p.stat().st_mtime)
    if not states:
        raise ValueError('Native trainer produced no training state')
    state = json.loads(states[-1].read_text())
    base = AutoModelForCausalLM.from_pretrained(request['base_model'],
        torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa').to('cuda:0')
    model = PeftModel.from_pretrained(base, str(output), local_files_only=True)
    changed = merged_change_count(model)
    if changed < 1 or state['global_step'] < 1:
        raise ValueError('Published dtype contains no numerical weight update')
    model = model.merge_and_unload()
    target = directory / 'model'
    target.mkdir(exist_ok=False)
    model.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(output), local_files_only=True).save_pretrained(target)
    result = {'algorithm': 'native-llamafactory-lora-sft', 'steps': state['global_step'],
        'world_size': request['world_size'],
        'weights_changed': True, 'merged_changed_elements': changed,
        'loss_history': {'train': [item for item in state.get('log_history', []) if 'loss' in item],
                         'eval': [item for item in state.get('log_history', []) if 'eval_loss' in item]}}
    (directory / 'training-result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()))
