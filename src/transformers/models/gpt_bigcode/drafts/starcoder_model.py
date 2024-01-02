from transformers import GPTBigCodeForCausalLM, GPTBigCodeConfig

from pathlib import Path
import json
import torch


if __name__ == "__main__":
    checkpoint_dir = "/admin/home/phuc_nguyen/.cache/huggingface/hub/models--HuggingFaceBR4--starcoder2_7b_4k_smol_data_580000/snapshots/92b6c25cab25f07c367bcc6d773635700a8a287d"
    checkpoint_dir = Path(checkpoint_dir)
    config = json.load(open(checkpoint_dir / "model_config.json"))
    
    import random

    import numpy as np
    
    seed = 42

    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    np.random.seed(seed)
    random.seed(seed)
    
    model_config = GPTBigCodeConfig(
        vocab_size=config["vocab_size"],
        n_positions=config["max_position_embeddings"],
        n_embd=config["hidden_size"],
        n_layer=config["num_hidden_layers"],
        n_head=config["num_attention_heads"],
        num_key_value_heads=config["num_kv_heads"],
        # NOTE: based on https://github.com/huggingface/brrr/blob/f569b93f80d03c626b24370d5ca4b1fe4f13fd76/brrr/models/fast/starcoder2.py#L194C16-L194C88
        n_inner=config.get("n_inner", 4 * config["hidden_size"]),
        activation_function=config["activation_function"],
        resid_pdrop=config["resid_pdrop"],
        embd_pdrop=config["embd_pdrop"],
        attn_pdrop=config["attn_pdrop"],
        layer_norm_epsilon=config["layer_norm_epsilon"],
        scale_attn_weights=config["scale_attn_weights"],
        bos_token_id=config["bos_token_id"],
        eos_token_id=config["eos_token_id"],
        attention_softmax_in_fp32=config["attention_softmax_in_fp32"],
        scale_attention_softmax_in_fp32=config["scale_attention_softmax_in_fp32"],
        multi_query=config["multi_query"],
        use_rotary_embeddings=config["use_rotary_embeddings"],
        # rotary_embedding_scale=brrr_model_config.rotary_embedding_scale, #TODO
        attention_window_size=config["sliding_window_size"],
    )

    model = GPTBigCodeForCausalLM._from_config(model_config, torch_dtype=torch.bfloat16)
    
    checkpoint_path = Path("/fsx/phuc/projects/starcoder/transformers-starcoder/src/transformers/models/gpt_bigcode/merged_checkpoint_reversed.pth")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to("cuda")
    
    # model = GPTBigCodeForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, device_map="cuda", config=config)
    
    assert 1 == 1
    
    from transformers import AutoTokenizer
    checkpoint = "bigcode/starcoder"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.eos_token_id = tokenizer.pad_token_id
    # inputs = tokenizer("123", return_tensors='pt').to("cuda")
    
    inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    print(f"outputs: {outputs}")
    
    print(tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False))

    # print([x for x in model.state_dict().keys()])

    # print("----------------------------------------------------------------\n")
    
    # for key, value in model.state_dict().items():
    #     print(key, value.shape)
