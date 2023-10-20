import argparse
import os
from pathlib import Path
import re

import torch
from transformers.models.gpt_bigcode.merge_fast_llm_checkpoint import merge_checkpoint
from transformers.models.gpt_bigcode import GPTBigCodeConfig, GPTBigCodeForCausalLM, GPTBigCodeModel


# The simple map of names for "automated" rules.
NAME_MAP = {
    "_mlp._layer_1": "mlp.c_fc",
    "_mlp._layer_2": "mlp.c_proj",
    "layer_norm_1": "ln_1",
    "layer_norm_2": "ln_2",
    # "attention.dense": "attn.c_proj",
    "self_attn.dense": "attn.c_proj",
    # "self_attention.query_key_value": "attn.c_attn",
}


def convert_fast_llm_checkpoint(state_dict, config):
    # The converted output model.
    output_state_dict = {}

    config = GPTBigCodeConfig(
        architectures=["GPTBigCodeLMHeadModel"],
        vocab_size=config["vocab_size"],
        n_positions=config["max_position_embeddings"],
        n_embd=config["hidden_size"],
        n_layer=config["num_layers"],
        n_head=config["num_attention_heads"],
        n_inner=config["ffn_hidden_size"],
        activation_function="gelu",  # TODO
        multi_query=True,  # TODO
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,  # TODO: can we remove these?
        eos_token_id=50256,
        attention_softmax_in_fp32=True,
        scale_attention_softmax_in_fp32=True,
        use_rotary_embeddings=config["use_rotary_embeddings"],
        rotary_embedding_scale=config["rotary_embedding_scale"],
        use_position_embeddings=config["use_position_embeddings"],
        attention_window_size=config["attention_window_size"]
    )

    # Truncate the word embeddings to the vocab-size
    word_embeddings = state_dict.pop("_layers.0._word_embeddings_weight")[:config.vocab_size, :]
    output_state_dict["transformer.wte.weight"] = word_embeddings
    # TODO: positional embeddings
    # Layer-0 is the word embeddings
    # Layers 1 to n_layer need to be re-mapped from 0 to n_layer-1.
    # _layers.{layer_index}.{op}.{w/b}

    # Concatenate QKV matrix
    for layer_index in range(1, config.n_layer + 1):
        for weight_or_bias in ["weight", "bias"]:
            query = state_dict.pop(f"_layers.{layer_index}.self_attn.query.{weight_or_bias}")
            key_value = state_dict.pop(f"_layers.{layer_index}.self_attn.key_value.{weight_or_bias}")
            output_state_dict[f"transformer.h.{layer_index - 1}.attn.c_attn.{weight_or_bias}"] = torch.cat([query, key_value], dim=0)
    
    # Extract the other ops
    layer_re = re.compile("_layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    for name, value in state_dict.items():
        m = layer_re.match(name)

        # The index of the layer.
        layer_index = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # Final layernorm
        if op_name == "final_layernorm":
            assert layer_index == config.n_layer + 1
            output_state_dict[f"transformer.ln_f.{weight_or_bias}"] = value
        else:
            output_state_dict[f"transformer.h.{layer_index-1}.{NAME_MAP[op_name]}.{weight_or_bias}"] = value

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_state_dict["lm_head.weight"] = word_embeddings

    return output_state_dict, config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        help="Path to the experiment directory",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Path where the converted model is saved"
    )
    args = parser.parse_args(argv)

    state_dict, config = merge_checkpoint(
        args.checkpoint_dir,
        dummy_experiment_dir="/toolkit_infiniband_example_checkpoints/ngc_checkpoints/sc2_ablations/dev_1B_repo_context_Random_tp4_pp2_8k_8k_2023_10_19_18_40_11/"
    )
    
    output_state_dict, output_config = convert_fast_llm_checkpoint(state_dict, config)
    
    print("Saving config")
    save_dir = args.save_dir or args.checkpoint_dir / "converted"
    output_config.save_pretrained(save_dir)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(save_dir, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)
    print(f'Done!')


if __name__ == "__main__":
    main()
