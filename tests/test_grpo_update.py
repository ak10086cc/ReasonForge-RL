"""A tiny, synthetic algorithm fixture; never reported as a GSM8K experiment."""
import torch
import copy
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, set_seed
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from reasonforge.modeling import adapter_state, weight_delta
from reasonforge.grpo import ResumeSafeGRPOTrainer


def test_real_grpo_optimizer_updates_adapter(tmp_path):
    set_seed(42)
    vocabulary = {"<pad>":0, "<eos>":1, "<bos>":2, "<unk>":3, **{f"t{i}":i for i in range(4,64)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
                                        bos_token="<bos>", unk_token="<unk>", padding_side="left")
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                       eos_token_id=1, pad_token_id=0, bos_token_id=2))
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    before = adapter_state(model)
    initial = copy.deepcopy(model)
    observed = []

    def token_reward(completions, **kwargs):
        # A deterministic textual reward with group variation tests the actual
        # rollout -> advantage -> gradient -> optimizer path, without gold labels.
        scores = [sum(ord(c) for c in completion) / 100 for completion in completions]
        observed.extend(scores)
        return scores

    trainer = ResumeSafeGRPOTrainer(model=model, processing_class=tokenizer,
                          args=GRPOConfig(output_dir=str(tmp_path), max_steps=1,
                                          per_device_train_batch_size=1, gradient_accumulation_steps=4,
                                          num_generations=4, max_completion_length=8,
                                          beta=.04, loss_type="grpo", learning_rate=.001,
                                          bf16=False, use_cpu=not torch.cuda.is_available(),
                                          logging_steps=1, save_strategy="steps", save_steps=1, report_to="none",
                                          disable_tqdm=True, use_vllm=False, seed=42),
                          train_dataset=Dataset.from_list([{"prompt":"t4 t5"}, {"prompt":"t6 t7"}]),
                          reward_funcs=[token_reward])
    trainer.train()
    assert len(observed) == 4 and len(set(observed)) > 1
    assert weight_delta(before, adapter_state(trainer.model))["parameters_changed"]
    restored = ResumeSafeGRPOTrainer(model=initial, processing_class=tokenizer, args=trainer.args,
                                     train_dataset=trainer.train_dataset, reward_funcs=[token_reward])
    restored._load_from_checkpoint(str(tmp_path/"checkpoint-1"))
    from reasonforge.modeling import state_hash
    assert state_hash(adapter_state(restored.model)) == state_hash(adapter_state(trainer.model))
    assert not any(p.requires_grad for n,p in restored.model.named_parameters() if ".ref." in n)
