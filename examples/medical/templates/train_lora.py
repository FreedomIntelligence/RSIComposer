"""Optional minimal LoRA SFT candidate. Host does not require this algorithm."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-tokens', type=int, default=2048)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model
    from training_support import sft_example, tensor_collator
    if Path(args.output).exists():
        raise ValueError('Use a fresh output directory')
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(line) for line in Path(args.train).read_text().splitlines() if line.strip()]
    dataset = [sft_example(tokenizer, [{'role': 'system', 'content': 'Please answer only.'},
        {'role': 'user', 'content': row['question'] + ('\n' + '\n'.join(
            key + '. ' + value for key, value in row['options'].items())
            if 'Answer Choices:' not in row['question'] else '')}], row['gold'], max_tokens=args.max_tokens)
        for row in rows]
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16,
        device_map='auto', local_files_only=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16,
        lora_alpha=32, lora_dropout=.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=args.output,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=1,
        learning_rate=1e-4, logging_steps=1, save_strategy='no', bf16=True,
        report_to=[], remove_unused_columns=False), train_dataset=dataset,
        data_collator=tensor_collator(int(tokenizer.pad_token_id)))
    trained = trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    Path(args.output, 'training-receipt.json').write_text(json.dumps({
        'authority': 'candidate_telemetry_not_host_attestation', 'optimizer_steps': trainer.state.global_step,
        'loss': trained.training_loss, 'examples': len(dataset),
        'medical_improvement_verified': False}))


if __name__ == '__main__':
    main()
