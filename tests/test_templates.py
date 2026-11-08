import pytest
from reasonforge.prompts import clean_solution, render_prompt, sft_record, grpo_record
from reasonforge.modeling import load_tokenizer
from reasonforge.utils import load_config


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(load_config("configs/smoke.yaml"))


def test_solution_and_prompt_alignment(tokenizer):
    row = {"question": "Unique question about 9831 widgets?", "answer": "There are 2+3=<<2+3=5>>5.\n#### 5",
           "ground_truth": "5", "question_id": "official_train:0", "split": "train"}
    sft, grpo = sft_record(row, tokenizer), grpo_record(row, tokenizer)
    assert sft["prompt"] == grpo["prompt"] == render_prompt(tokenizer, row["question"])
    assert "There are" not in grpo["prompt"]
    assert "Final Answer: 5" not in grpo["prompt"]
    assert "<<" not in sft["completion"] and "2+3=5" in sft["completion"]
    assert sft["completion"].endswith(tokenizer.eos_token)
    assert clean_solution(row["answer"]).endswith("Final Answer: 5")


def test_actual_trl_collator_masks_prompt_and_padding():
    import torch
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(pad_token_id=0)
    batch = collator([{"input_ids": [11,12,13,14], "labels": [-100,-100,13,14]},
                      {"input_ids": [11,12,15], "labels": [-100,-100,15]}])
    assert torch.equal(batch["labels"], torch.tensor([[-100,-100,13,14],[-100,-100,15,-100]]))


def test_actual_sfttrainer_preparation_masks_only_prompt(tokenizer, tmp_path):
    from datasets import Dataset
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import SFTConfig, SFTTrainer
    from reasonforge.sft import audit_loss_mask
    row = {"question": "What is 2 plus 3?", "answer": "2 plus 3 equals 5.\n#### 5"}
    record = sft_record(row, tokenizer)
    # Small random architecture is a unit fixture, never an experimental baseline.
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=16,
                                       intermediate_size=32, num_hidden_layers=1,
                                       num_attention_heads=2, num_key_value_heads=1))
    trainer = SFTTrainer(model=model, processing_class=tokenizer,
                         args=SFTConfig(output_dir=str(tmp_path), max_length=512, completion_only_loss=True,
                                        shuffle_dataset=False, bf16=False, use_cpu=True, report_to="none"),
                         train_dataset=Dataset.from_list([record]))
    audit = audit_loss_mask(trainer, record["prompt"], tokenizer)
    assert audit["passed"] and audit["supervised_tokens"] > 0
