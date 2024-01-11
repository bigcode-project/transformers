import argparse
import os
from pathlib import Path
import re

import torch
from transformers.models.gpt_bigcode.merge_fast_llm_checkpoint import merge_checkpoint
from transformers.models.gpt_bigcode import GPTBigCodeConfig


def convert_fast_llm_checkpoint(state_dict, config, set_attn_dense_bias_zero, set_mlp_2_bias_zero, version=1):
    if set_attn_dense_bias_zero:
        print("Will set attention output layer biases to zero")
    if set_mlp_2_bias_zero:
        print("Will set MLP layer-2 biases to zero")
    # The converted output model.
    output_state_dict = {}
    if "window_size" in config:
        attention_window_size = config["window_size"]
    else:
        attention_window_size = config.get("attention_window_size", None)

    config = GPTBigCodeConfig(
        architectures=["GPTBigCodeLMHeadModel"],
        vocab_size=config["vocab_size"],
        n_positions=config["max_position_embeddings"],
        n_embd=config["hidden_size"],
        n_layer=config["num_layers"],
        n_head=config["num_attention_heads"],
        head_groups=config.get("head_groups", None),
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
        bos_token_id=0,  # TODO: can we remove these?
        eos_token_id=0,
        attention_softmax_in_fp32=True,
        scale_attention_softmax_in_fp32=True,
        use_rotary_embeddings=config["use_rotary_embeddings"],
        rotary_embedding_scale=config["rotary_embedding_scale"],
        use_position_embeddings=config["use_position_embeddings"],
        attention_window_size=attention_window_size
    )

    # Truncate the word embeddings to the vocab-size
    u="_" if version==0 else ""
    word_embeddings = state_dict.pop(f"{u}layers.0.{u}word_embeddings_weight")[:config.vocab_size, :]
    output_state_dict["transformer.wte.weight"] = word_embeddings
    if config.use_position_embeddings:
        output_state_dict["transformer.wpe.weight"] = state_dict.pop(f"{u}layers.0.{u}position_embeddings_weight")

    # Layer-0 is the word/position embeddings
    # Layers 1 to n_layer need to be re-mapped from 0 to n_layer-1.
    # _layers.{layer_index}.{op}.{w/b}

    # Concatenate QKV matrix
    for layer_index in range(1, config.n_layer + 1):
        for weight_or_bias in ["weight", "bias"]:
            query = state_dict.pop(f"{u}layers.{layer_index}.self_attn.query.{weight_or_bias}")
            key_value = state_dict.pop(f"{u}layers.{layer_index}.self_attn.key_value.{weight_or_bias}")
            output_state_dict[f"transformer.h.{layer_index - 1}.attn.c_attn.{weight_or_bias}"] = torch.cat([query, key_value], dim=0)

    # The simple map of names for "automated" rules.
    name_map = {
        f"{u}mlp.{u}layer_1": "mlp.c_fc",
        f"{u}mlp.{u}layer_2": "mlp.c_proj",
        "layer_norm_1": "ln_1",
        "layer_norm_2": "ln_2",
        # "attention.dense": "attn.c_proj",
        "self_attn.dense": "attn.c_proj",
        # "self_attention.query_key_value": "attn.c_attn",
    }
    # Extract the other ops
    layer_re = re.compile(f"{u}layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    for name, value in state_dict.items():
        m = layer_re.match(name)
        assert m is not None, f"Invalid layer name: {name}"

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
        # Bias was not used in training for InputParallel layers
        elif op_name == "self_attn.dense" and weight_or_bias == "bias" and set_attn_dense_bias_zero:
            output_state_dict[f"transformer.h.{layer_index-1}.{name_map[op_name]}.{weight_or_bias}"] = torch.zeros_like(value)
        # MLP layer-2 is also InputParallel
        elif op_name == f"{u}mlp.{u}layer_2" and weight_or_bias == "bias" and set_mlp_2_bias_zero:
            output_state_dict[f"transformer.h.{layer_index-1}.{name_map[op_name]}.{weight_or_bias}"] = torch.zeros_like(value)
        else:
            output_state_dict[f"transformer.h.{layer_index-1}.{name_map[op_name]}.{weight_or_bias}"] = value

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
    parser.add_argument(
        "--set_attn_dense_bias_zero",
        action='store_true',
        default=False,
        help="Set the attention output layer bias to zero and ignore the value from the checkpoint. Shouldn't be used except to fix a bug from training."
    )
    parser.add_argument(
        "--set_mlp_2_bias_zero",
        action='store_true',
        default=False,
        help="Set the MLP second layer bias to zero and ignore the value from the checkpoint. Shouldn't be used except to fix a bug from training."
    )
    
    args = parser.parse_args(argv)

    state_dict, config = merge_checkpoint(
        args.checkpoint_dir,
        dummy_experiment_dir=None
    )
    
    output_state_dict, output_config = convert_fast_llm_checkpoint(state_dict, config, args.set_attn_dense_bias_zero, args.set_mlp_2_bias_zero)
    
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
