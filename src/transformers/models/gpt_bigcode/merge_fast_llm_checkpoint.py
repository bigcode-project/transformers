import re
from pathlib import Path

import torch
from safetensors import safe_open
from collections import defaultdict

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


def transform_paths(paths):
    path_objs = [Path(p) for p in paths]
    common_path_prefix = Path(commonprefix(path_objs)).parent

    final_paths = {}
    for path in path_objs:
        relative_path = str(path.relative_to(common_path_prefix))
        dot_path = relative_path.replace('/', '.')
        
        weight_replaced = re.sub(r'model_weight_pp-rank-0-of-1_tp-rank-(\d)-of-4', r'weight.\1', dot_path)
        bias_replaced = re.sub(r'model_bias_pp-rank-0-of-1_tp-rank-(\d)-of-4', r'bias.\1', weight_replaced)
        cleaned_path = bias_replaced.replace('.safetensors', '')

        final_paths[cleaned_path] = str(path)

    return final_paths

def group_and_sort_paths(paths):
    grouped_paths = defaultdict(list)

    for key, path in paths.items():
        try:
            module_name, shard_number = key.rsplit('.', 1)
            grouped_paths[module_name].append((int(shard_number), path))
        except ValueError:
            # NOTE: these are layer norm's weight and biases
            # so it don't have shard number
            grouped_paths[key].append(path)

    # Remove any entries with empty lists
    grouped_paths = {k: v for k, v in grouped_paths.items() if v}

    # NOTE: Sort paths in each group
    # module: [(4, path), (0, path), (3, path) ...] -> module: [(0, path), (1, path), (2, path) ...]
    sorted_grouped_paths = {module: sorted(paths, key=lambda x: x[0])
                            for module, paths in grouped_paths.items()}

    return sorted_grouped_paths
            
def merge_checkpoints(paths):
    def find_corresponding_dim(name):
        for key, value in MERGE_DIM_MAPPING.items():
            if key in name:
                return value
        return None

    model_states = {}
    for state_key, path in paths.items():
        model_states[state_key] = {}
        for shard_id, _path in enumerate(path):
            checkpoint_path = _path[1] if isinstance(_path, tuple) else _path
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    data = f.get_tensor(key)
                    model_states[state_key][shard_id] = data
        
        tensor_list = [tensor for _, tensor in sorted(model_states[state_key].items())]
        merge_dim = find_corresponding_dim(state_key)

        if len(tensor_list) > 1:
            model_states[state_key] = torch.cat(tensor_list, dim=merge_dim)
        else:
            # NOTE: these are biases
            model_states[state_key] = tensor_list[0]
    return model_states

def remap_keys(target_dict):
    key_mapping = {
        'model.final_layer_norm.pp_block.model_weight': 'transformer.ln_f.weight',
        'model.final_layer_norm.pp_block.model_bias': 'transformer.ln_f.bias',
        'model.token_embeddings.pp_block.token_embedding.weight': 'transformer.wte.weight'
    }

    def get_new_key(key):
        if 'model.decoder' in key and 'pp_block' in key:
            parts = key.split('.')
            block_number = parts[2]
            component_parts = parts[4:]
            component = '.'.join(component_parts)
            new_component = BRRR_TFMS_NAME_MAPPING.get(component, component)
            return f"transformer.h.{block_number}.{new_component}"
        else:
            return key_mapping.get(key, key)

    new_dict = {get_new_key(key): value for key, value in target_dict.items()}
    new_dict["lm_head.weight"] = new_dict.get("transformer.wte.weight", new_dict.get("lm_head.weight"))
    
    return new_dict

def merge_checkpoint(checkpoint_dir: Path):
    """Load a checkpoint from the BRRR format and merge tensor parallel shards."""
    checkpoint_paths = get_safetensor_checkpoint_paths(checkpoint_dir)
    paths = transform_paths(checkpoint_paths)
    paths = group_and_sort_paths(paths)
    model_states = merge_checkpoints(paths)
    model_states = remap_keys(model_states)

    return model_states
