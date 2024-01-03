import re
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import yaml

import re
from os.path import commonprefix


MERGE_DIM_MAPPING = {
    "ff.c_fc.bias": 0,
    "token_embedding": 0, # row linear parallel
    "c_fc": 0, # column linear parallel
    "c_proj": 1, # row linear parallel
    # NOTE: weird
    "query_key_value": 0, # row linear parallel
    "dense": 1, # row linear parallel
}

BRRR_TFMS_NAME_MAPPING = {
    'ln_1.model_weight': 'ln_1.weight',
    'ln_1.model_bias': 'ln_1.bias',
    'ln_2.model_weight': 'ln_2.weight',
    'ln_2.model_bias': 'ln_2.bias',
    'attn.query_key_value.weight': 'attn.c_attn.weight',
    'attn.query_key_value.bias': 'attn.c_attn.bias',
    'attn.dense.weight': 'attn.c_proj.weight',
    'attn.dense.model_bias': 'attn.c_proj.bias',
    'ff.c_fc.weight': 'mlp.c_fc.weight',
    'ff.c_fc.bias': 'mlp.c_fc.bias',
    'ff.c_proj.weight': 'mlp.c_proj.weight',
    'ff.c_proj.model_bias': 'mlp.c_proj.bias'
}

def get_safetensor_checkpoint_paths(checkpoint_dir: Path):
    model_dir = checkpoint_dir / "model" / "model"
    safetensor_files = []

    for file_path in model_dir.rglob("*.safetensors"):
        if file_path.is_file():
            safetensor_files.append(file_path.absolute())

    return safetensor_files

def merge_checkpoint(checkpoint_dir: Path, dummy_experiment_dir=None):
    """Load a fast-llm checkpoint and merge the data, tensor, and pipeline-parallel shards"""
    # checkpoint_dir=experiment_dir/checkpoints/{iteration}
    # experiment_dir = "~/.cache/huggingface/hub/models--HuggingFaceBR4--starcoder2_7b_4k_smol_data_580000"
    # experiment_dir = checkpoint_dir.parent.parent
    
    # NOTE: use the checkpoint format from https://huggingface.co/HuggingFaceBR4/starcoder2_7b_4k_smol_data_580000/tree/main/model/model/token_embeddings/pp_block/token_embedding
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_paths = get_safetensor_checkpoint_paths(checkpoint_dir)
    config = yaml.safe_load((checkpoint_dir / "config.yaml").read_text())

    def convert_paths_to_dict(paths):
        path_objs = [Path(p) for p in paths]
        common_path_prefix = Path(commonprefix(path_objs)).parent
        path_dict = {str(p.relative_to(common_path_prefix)): str(p) for p in path_objs}
        return path_dict
    
    paths = convert_paths_to_dict(checkpoint_paths)
    
    def convert_slashes_to_dots(input_dict):
        converted_dict = {}
        for key, value in input_dict.items():
            modified_key = key.replace('/', '.')
            converted_dict[modified_key] = value

        return converted_dict
    
    paths = convert_slashes_to_dots(paths)
    
    def replace_patterns(paths):
        new_paths = {}
        for key, value in paths.items():
            new_key = re.sub(r'model_weight_pp-rank-0-of-1_tp-rank-(\d)-of-4', r'weight.\1', key)
            new_key = re.sub(r'model_bias_pp-rank-0-of-1_tp-rank-(\d)-of-4', r'bias.\1', new_key)
            new_paths[new_key] = value
        return new_paths
    
    paths = replace_patterns(paths)
    
    def remove_safetensors_extension(paths):
        new_paths = {}
        for key, value in paths.items():
            new_key = key.replace('.safetensors', '')
            new_paths[new_key] = value
        return new_paths
    
    paths = remove_safetensors_extension(paths)

    from collections import defaultdict
    grouped_paths = defaultdict(list)
    for key, path in paths.items():
        try:
            module_name, shard_number = key.rsplit('.', 1)
            grouped_paths[module_name].append((int(shard_number), path))
        except:
            # NOTE: these are layer norm's weight, bias
            # or other module biases, which are small, so brrr doesn't split them
            print(f"skipped {key}, {path}")
            grouped_paths[key].append(path)

    def remove_keys_with_empty_lists(input_dict):
        filtered_dict = {key: value for key, value in input_dict.items() if value}
        return filtered_dict
    
    grouped_paths = remove_keys_with_empty_lists(grouped_paths)
    
    # TODO(xrsrke): it merged paths for bias and weight in the same group => wrong
    # sorted_grouped_paths = {module: sorted(paths, key=lambda x: x[0]) for module, paths in grouped_paths.items()}
    sorted_grouped_paths = {module: sorted(paths, key=lambda x: x[0], reverse=True) for module, paths in grouped_paths.items()}
    paths = sorted_grouped_paths
    
    from safetensors import safe_open
    
    assert 1 == 1
    
    def find_corresponding_dim(name):
        for key, value in MERGE_DIM_MAPPING.items():
            if key in name:
                return value
        return None

    _model_states = {}
    for state_key, path in paths.items():
        _model_states[state_key] = {}
        for shard_id, _path in enumerate(path):
            checkpoint_path = _path[1] if isinstance(_path, tuple) else _path
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    data = f.get_tensor(key)
                    _model_states[state_key][shard_id] = data
        
        tensor_list = [tensor for _, tensor in sorted(_model_states[state_key].items())]
        merge_dim = find_corresponding_dim(state_key)
        print(f"trying to merge: {state_key}")

        if len(tensor_list) > 1:
            try:
                _model_states[state_key] = torch.cat(tensor_list, dim=merge_dim)
            except:
                print(f"skipped {state_key}, {[x.shape for x in tensor_list]}")
        else:
            # NOTE: these are biases
            _model_states[state_key] = tensor_list[0]

    def remap_keys(target_dict):
        new_dict = {}
        for key, value in target_dict.items():
            parts = key.split('.')

            if 'model.decoder' in key and 'pp_block' in key:
                block_number = parts[2]
                component_parts = parts[4:]
                component = '.'.join(component_parts)

                new_component = BRRR_TFMS_NAME_MAPPING.get(component, component)
                new_key = f"transformer.h.{block_number}.{new_component}"
                new_dict[new_key] = value
            elif key == 'model.final_layer_norm.pp_block.model_weight':
                new_dict['transformer.ln_f.weight'] = value
            elif key == 'model.final_layer_norm.pp_block.model_bias':
                new_dict['transformer.ln_f.bias'] = value

            elif key == 'model.token_embeddings.pp_block.token_embedding.weight':
                new_dict['transformer.wte.weight'] = value

        return new_dict

    _model_states = remap_keys(_model_states)
    _model_states["lm_head.weight"] = _model_states["transformer.wte.weight"]
    
    for key, value in _model_states.items():
        if isinstance(value, torch.Tensor):
            print(f"key: {key}, value: {value.shape} \n")
        else:
            print(f"skipped key: {key}, shape: {[x.shape for x in value.values()]} \n")
    
    print("saving merged checkpoint...")
    
    torch.save(_model_states, './merged_checkpoint_reversed.pth')
    
    print("done")
