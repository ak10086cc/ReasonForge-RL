import copy
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, set_seed
from peft import LoraConfig, PeftModel, get_peft_model
from reasonforge.modeling import adapter_state, state_hash, weight_delta


def test_lora_real_update_save_reload(tmp_path):
    set_seed(42)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1))
    fresh_base = copy.deepcopy(base)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    before = adapter_state(model)
    ids = torch.tensor([[1,2,3,4]])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(ids, labels=torch.tensor([[-100,-100,3,4]])).loss.backward()
    optimizer.step()
    after = adapter_state(model)
    assert weight_delta(before, after)["parameters_changed"]
    model.save_pretrained(tmp_path, save_embedding_layers=False)
    reloaded = PeftModel.from_pretrained(fresh_base, tmp_path, is_trainable=True)
    assert state_hash(adapter_state(reloaded)) == state_hash(after)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, reloaded(ids).logits)
