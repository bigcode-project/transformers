from transformers import GPTBigCodeForCausalLM, GPTBigCodeConfig

from pathlib import Path
import json


if __name__ == "__main__":
    checkpoint_dir = "/admin/home/phuc_nguyen/.cache/huggingface/hub/models--HuggingFaceBR4--starcoder2_7b_4k_smol_data_580000/snapshots/92b6c25cab25f07c367bcc6d773635700a8a287d"
    checkpoint_dir = Path(checkpoint_dir)
    config = json.load(open(checkpoint_dir / "model_config.json"))
    
    model_config = GPTBigCodeConfig(
        vocab_size=config["vocab_size"],
        n_positions=config["max_position_embeddings"],
        n_embd=config["hidden_size"],
        n_layer=config["num_hidden_layers"],
        n_head=config["num_attention_heads"],
        num_key_value_heads=config["num_kv_heads"],
        n_inner=config["hidden_size"],
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

    model = GPTBigCodeForCausalLM._from_config(model_config)
