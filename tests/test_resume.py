"""Check real optimizer/RNG checkpoint resume against uninterrupted training."""
import copy
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, TrainerCallback, set_seed
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTConfig, SFTTrainer
import torch
from reasonforge.modeling import adapter_state


def test_sft_optimizer_resume_matches_uninterrupted(tmp_path):
    set_seed(42)
    vocabulary = {"<pad>":0,"<eos>":1,"<unk>":2, **{f"t{i}":i for i in range(3,64)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    base_for_reload = copy.deepcopy(base)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                           target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM"))
    initial = copy.deepcopy(model)
    dataset = Dataset.from_list([{"input_ids":[3,4,5,1],"labels":[-100,-100,5,1]},
                                 {"input_ids":[6,7,8,1],"labels":[-100,-100,8,1]}])

    def trainer(m, name, callbacks=None):
        return SFTTrainer(model=m, processing_class=tokenizer, train_dataset=dataset, callbacks=callbacks,
                          args=SFTConfig(output_dir=str(tmp_path/name), max_steps=2, per_device_train_batch_size=1,
                                         learning_rate=.001, lr_scheduler_type="constant", save_steps=1,
                                         bf16=False, use_cpu=True, report_to="none", disable_tqdm=True,
                                         shuffle_dataset=False, dataset_kwargs={"skip_prepare_dataset":True}, seed=42))

    full = trainer(model, "full")
    full.train()
    expected = adapter_state(full.model)

    class InterruptAfterFirstStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_save = True
                control.should_training_stop = True

    interrupted = trainer(initial, "interrupted", [InterruptAfterFirstStep()])
    interrupted.train()
    checkpoint = tmp_path/"interrupted/checkpoint-1"
    assert (checkpoint/"optimizer.pt").is_file() and (checkpoint/"scheduler.pt").is_file()
    reloaded = PeftModel.from_pretrained(base_for_reload, checkpoint, is_trainable=True)
    resumed = trainer(reloaded, "resumed")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    actual = adapter_state(resumed.model)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=1e-5, atol=1e-7)
