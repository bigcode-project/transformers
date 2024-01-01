import re
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import yaml


def get_all_checkpoint_paths(experiment_path):
    checkpoints = (Path(experiment_path) / "checkpoints").glob("*")
    # Sort checkpoints by iteration number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.name))
    return [get_checkpoint_paths(checkpoint) for checkpoint in checkpoints]


def get_checkpoint_paths(checkpoint_dir: Path):
    # model/model
    return [c_name for c_name in checkpoint_dir.glob("*") if re.match(r"\d+", c_name.name)]

def get_safetensor_checkpoint_paths(checkpoint_dir: Path):
    model_dir = checkpoint_dir / "model" / "model"  # Targeting the specific directory
    safetensor_files = []

    for file_path in model_dir.rglob("*.safetensors"):  # Looking for files with .safetensors extension
        if file_path.is_file():  # Ensure it's a file
            safetensor_files.append(file_path.absolute())  # Adding the absolute path of the file

    return safetensor_files


def extract_stage_shards(state):
    # Extract the weight shard and split it into the stage shards
    # Reproduce the split done in MultiStageModelBase.setup
    total_shard_size = sum(state['stage_shard_sizes'])
    if len(state['shard'].shape) == 1:
        # Flat buffer
        weight_shard = state['shard'][:total_shard_size]
    elif len(state['shard'].shape) == 2:
        # 2D buffer
        weight_shard = state['shard'][0]
    else:
        raise ValueError(f"Unrecognized buffer shape {state['shard'].shape}")
    return weight_shard.split(state['stage_shard_sizes'])


def extract_individual_weights(merged_stage_shard, stage_content):
    # Get individual weights from shards that are merged across data-parallel
    weights_numel = [np.prod(weight_meta['shape']) for weight_meta in stage_content]
    weights = merged_stage_shard[:sum(weights_numel)].split(weights_numel)
    return [weight.reshape(weight_meta['shape']) for weight, weight_meta in zip(weights, stage_content)]


def concatenate_tp_shards(stage_tp_shards, stage_content):
    # Concatenate the tp-shards in a given stage
    # Stage_tp_shards: contains the individual weight shards for each rank
    # [[weight1, weight2, ...] for rank in range(tp_size)]
    concatenated_weights = []
    # Concatenate each individual weight along their TP dimension if they have one.
    for weight_tp_shards, weight_meta in zip(zip(*stage_tp_shards), stage_content):
        if weight_meta["tensor_parallel_dim"] is not None:
            weight = torch.cat(weight_tp_shards, dim=weight_meta["tensor_parallel_dim"])
        else:
            weight = weight_tp_shards[0]
        concatenated_weights.append(weight)
    return concatenated_weights


def merge_checkpoint(checkpoint_dir: Path, dummy_experiment_dir=None):
    """Load a fast-llm checkpoint and merge the data, tensor, and pipeline-parallel shards"""
    # checkpoint_dir=experiment_dir/checkpoints/{iteration}
    # experiment_dir = "~/.cache/huggingface/hub/models--HuggingFaceBR4--starcoder2_7b_4k_smol_data_580000"
    # experiment_dir = checkpoint_dir.parent.parent
    
    # NOTE: use the checkpoint format from https://huggingface.co/HuggingFaceBR4/starcoder2_7b_4k_smol_data_580000/tree/main/model/model/token_embeddings/pp_block/token_embedding
    # where experiment_dir = checkpoint_dir
    # checkpoint_paths = get_checkpoint_paths(checkpoint_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_paths = get_safetensor_checkpoint_paths(checkpoint_dir)
    config = yaml.safe_load((checkpoint_dir / "config.yaml").read_text())
    
    # def path2tfm_name(path):
    #     name = path
    #     # remove `_pp-rank*` and what comes after
    #     name = name.split("_pp-rank-")[0]

    #     # remove `.safetensors`
    #     name = name.split(".safetensors")[0]

    #     # remove base path
    #     name = name.split(str(checkpoint_path) + "/model/")[1]

    #     # "/" -> "."
    #     name = name.replace("/", ".")

    #     # remove "model." prefix if lm_head
    #     if ".lm_head." in name:
    #         name = name[len("model.") :]

    #     # remove ".pp_block."
    #     name = name.replace(".pp_block.", ".")

    #     # apply mapping
    #     name = apply_mappings(name, BRRR_TRFRS_NAME_MAPPING)
    #     # print(name, path)

    #     # skip buffers
    #     if name.endswith(".model_inv_freq"):
    #         continue
    #     return name

    # Load the states from all the ranks
    
    import re

    # def create_state_dict(paths):
    #     state_dict = {}
    #     for path in paths:
    #         # Break down the path and extract relevant parts
    #         parts = path.parts
    #         # Find the tp-rank part and extract the rank number
    #         tp_rank_match = re.search(r'tp-rank-(\d+)-of-\d+', str(path))
    #         if tp_rank_match:
    #             tp_rank = tp_rank_match.group(1)
    #         else:
    #             continue  # Skip if tp-rank is not found

    #         # Construct the key from the path segments
    #         key_segments = [part for part in parts if part not in ['model_weight', 'pp_block', 'model']]
    #         key = '.'.join(key_segments[-5:])  # Adjust the index as needed to capture the right segments
    #         key = key.replace('/', '.').replace('\\', '.') + '.' + tp_rank

    #         # Add to the dictionary
    #         state_dict[key] = path

    #     return state_dict
    
    # def create_state_dict(paths):
    #     state_dict = {}
    #     keyword_mapping = {
    #         'model_bias': 'bias',
    #         'model_weight': 'weight',
    #     }

    #     for path in paths:
    #         tp_rank_match = re.search(r'tp-rank-(\d+)-of-\d+', str(path))
    #         if tp_rank_match:
    #             tp_rank = tp_rank_match.group(1)
    #         else:
    #             continue  # Skip if tp-rank is not found

    #         file_name = path.stem

    #         for key_word, replacement in keyword_mapping.items():
    #             file_name = replacement

    #         key = '.'.join(path.parts[-5:-1]) + '.' + file_name + '.' + tp_rank  # Modify indices as needed
    #         state_dict[key] = path

    #     return state_dict


    # state_dict = create_state_dict(checkpoint_paths)
    
    from os.path import commonprefix

    def convert_paths_to_dict(paths):
        # Convert strings to Path objects
        path_objs = [Path(p) for p in paths]

        # Find the common path prefix
        common_path_prefix = Path(commonprefix(path_objs)).parent

        # Create a dictionary with the modified paths
        path_dict = {str(p.relative_to(common_path_prefix)): str(p) for p in path_objs}

        return path_dict
    
    paths = convert_paths_to_dict(checkpoint_paths)
    
    def convert_slashes_to_dots(input_dict):
        # Create a new dictionary to store the modified keys and values
        converted_dict = {}

        # Iterate over the items in the input dictionary
        for key, value in input_dict.items():
            # Replace all forward slashes in the key with dots
            modified_key = key.replace('/', '.')

            # Add the modified key and its corresponding value to the new dictionary
            converted_dict[modified_key] = value

        return converted_dict
    
    paths = convert_slashes_to_dots(paths)
    
    # def group_by_prefix(input_dict, depth=1):
    #     grouped_dict = {}
    #     for key, value in input_dict.items():
    #         # Split the key, extract the prefix based on the specified depth
    #         prefix = '.'.join(key.split('.')[:depth])
    #         # Append the item to the corresponding list in the dictionary
    #         grouped_dict.setdefault(prefix, []).append(value)
    #     return grouped_dict

    # def group_by_prefix_and_type(input_dict, prefix_depth):
    #     grouped_dict = {}
    #     for idx, (key, value) in enumerate(input_dict.items()):
    #         # Split the key and extract the prefix
    #         key_parts = key.split('.')
    #         prefix = '.'.join(key_parts[:prefix_depth])

    #         # Determine if the key is for a weight or bias
    #         if 'weight' in key_parts:
    #             prefix += '.weight'
    #         elif 'bias' in key_parts:
    #             prefix += '.bias'

    #         # Append the index and link to the corresponding list in the dictionary
    #         grouped_dict.setdefault(prefix, []).append((idx, value))
    #     return grouped_dict
    
    def replace_patterns(paths):
        new_paths = {}
        for key, value in paths.items():
            # Replace the pattern with 'weight.x' or 'bias.x'
            new_key = re.sub(r'model_weight_pp-rank-0-of-1_tp-rank-(\d)-of-4', r'weight.\1', key)
            new_key = re.sub(r'model_bias_pp-rank-0-of-1_tp-rank-(\d)-of-4', r'bias.\1', new_key)
            new_paths[new_key] = value
        return new_paths
    
    paths = replace_patterns(paths)
    
    def remove_safetensors_extension(paths):
        new_paths = {}
        for key, value in paths.items():
            # Remove the '.safetensors' from the key
            new_key = key.replace('.safetensors', '')
            new_paths[new_key] = value
        return new_paths
    
    paths = remove_safetensors_extension(paths)
        
    # NOTE: probably the merge checkpoint paths are wrong
    assert 1 == 1

    from collections import defaultdict
    grouped_paths = defaultdict(list)
    for key, path in paths.items():
        try:
            module_name, shard_number = key.rsplit('.', 1)
            # module_name, shard_number, _ = key.rsplit('.', 2)
            grouped_paths[module_name].append((int(shard_number), path))
        except:
            # NOTE: these are layer norm's weight, bias
            # or other module biases, which are small, so brrr doesn't split them
            print(f"skipped {key}, {path}")
            grouped_paths[key].append(path)

    def remove_keys_with_empty_lists(input_dict):
        # Using dictionary comprehension to filter out keys with empty lists
        filtered_dict = {key: value for key, value in input_dict.items() if value}
        return filtered_dict
    
    grouped_paths = remove_keys_with_empty_lists(grouped_paths)
    
    # TODO(xrsrke): it merged paths for bias and weight in the same group => wrong
    sorted_grouped_paths = {module: sorted(paths, key=lambda x: x[0]) for module, paths in grouped_paths.items()}
    paths = sorted_grouped_paths
    
    from safetensors import safe_open
    
    assert 1 == 1
    
    MERGE_DIM_MAPPING = {
        "ff.c_fc.bias": 0,
        "token_embedding": 0, # row linear parallel
        "c_fc": 1, # column linear parallel
        "c_proj": 0, # row linear parallel
        # NOTE: weird
        "query_key_value": 0, # row linear parallel
        "dense": 1, # row linear parallel
    }
    
    def find_corresponding_dim(name):
        """
        Searches the MERGE_DIM_MAPPING for a key that is a substring of the given name.
        Returns the corresponding dimension if found, otherwise None.
        """
        for key, value in MERGE_DIM_MAPPING.items():
            if key in name:
                return value
        return None

    # path_demo = list(grouped_paths.values())[0]
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
        # if state_key == "28.pp_block.attn.query_key_value.weight" or state_key == "2.pp_block.attn.query_key_value.weight":
        #     assert 1 == 1
        #     continue
            
        # if state_key == "31.pp_block.ff.c_fc.weight" or state_key == "5.pp_block.attn.query_key_value.weight":
        #     continue
        
        # if state_key == "5.pp_block.ff.c_fc.weight":
        #     continue
        
        # if state_key == "17.pp_block.attn.query_key_value.weight":
        #     continue
        
        # if state_key == "0.pp_block.ff.c_fc.weight" or state_key == "20.pp_block.attn.query_key_value.weight":
        #     continue
        
        # if state_key == "18.pp_block.ff.c_fc.weight":
        #     continue
        
        if len(tensor_list) > 1:
            try:
                _model_states[state_key] = torch.cat(tensor_list, dim=merge_dim)
            except:
                print(f"skipped {state_key}, {[x.shape for x in tensor_list]}")
        else:
            # NOTE: these are biases
            _model_states[state_key] = tensor_list[0]
    
    assert 1 == 1
    
    for key, value in _model_states.items():
        if isinstance(value, torch.Tensor):
            print(f"key: {key}, value: {value.shape} \n")
        else:
            print(f"skipped key: {key}, shape: {[x.shape for x in value.values()]} \n")
    
    
    # states = {
    #     int(c_name.name): torch.load(c_name)
    #     for c_name in tqdm(checkpoint_paths)
    # }
    # # num_stages = len(states[0]["stages"])
    
    # # tensor_parallel = config["tensor_parallel"]
    # # data_parallel_size = int(config["world_size"] / (tensor_parallel * config["pipeline_parallel"]))
    # tensor_parallel_size = config["parallelism"]["tp"]
    # pipeline_parallel_size = config["parallelism"]["pp"]
    # data_parallel_size = config["parallelism"]["dp"]

    # if dummy_experiment_dir is not None:
    #     # Use the meta from the dummy checkpoint, and the shard from the actual checkpoint
    #     dummy_checkpoint_paths = get_all_checkpoint_paths(dummy_experiment_dir)
    #     dummy_states = {
    #         int(c_name.name): torch.load(c_name)
    #         for c_name in tqdm(dummy_checkpoint_paths[-1])
    #     }
    #     for rank, state in dummy_states.aitems():
    #         state['shard'] = states[rank]['shard']
    #     states = dummy_states

    # # Gather the data-parallel shards
    # # {tp_rank: [[stage_0_shard_0, stage_0_shard_1, ...], [stage_1_shard_0, ...], ...]}
    # # {tp_rank: [{fsdp_rank: shard}, ...]}
    # fsdp_shards = {
    #     i: [[None for _ in range(data_parallel_size)] for _ in range(pipeline_parallel_size)]
    #     for i in range(tensor_parallel_size)
    # }
    
    # for rank, state in states.items():
    #     on_device_stage_shards = extract_stage_shards(state)
    #     on_device_stage_indices = [i for (i, stage_meta) in enumerate(state["stages"]) if stage_meta["on_device"]]
    #     for stage_index, stage_shard in zip(on_device_stage_indices, on_device_stage_shards):
    #         stage_meta = state["stages"][stage_index]
    #         # fsdp_shards[stage_meta["tp_rank"]][stage_index].append((stage_meta, stage_shard))
    #         fsdp_shards[stage_meta["tp_rank"]][stage_index][stage_meta["fsdp_rank"]] = stage_shard
    
    # # Concatenate the data-parallel shards
    # # and get individual weights
    # dp_concatenated_shards = {
    #     tp_rank: [
    #         extract_individual_weights(
    #             torch.cat(stage_shards, dim=0),
    #             states[0]["stages"][stage_index]['content']
    #         )
    #         for stage_index, stage_shards in enumerate(fsdp_shards[tp_rank])
    #     ]
    #     for tp_rank in range(config["tensor_parallel"])
    # }

    # # In the tensor-parallel case, concatenate the TP tensors along their TP dimensions.
    # tp_concatenated_shards = []
    # for stage_index, stage_tp_shards in enumerate(zip(*(dp_concatenated_shards[i] for i in range(tensor_parallel)))):
    #     stage_content = states[0]["stages"][stage_index]["content"]
    #     tp_concatenated_shards.append(concatenate_tp_shards(stage_tp_shards, stage_content))

    # # In the pipeline-parallel case, merge the stages
    # state_dict = {
    #     weight_meta["name"]: weight
    #     for stage_meta, stage_weights in zip(states[0]["stages"], tp_concatenated_shards)
    #     for weight_meta, weight in zip(stage_meta["content"], stage_weights)
    # }

    # print(f"Total number of parameters: {sum([weight.numel() for weight in state_dict.values()])}")
    # return state_dict, config


# if __name__ == "__main__":
#     merge_checkpoint("/toolkit_infiniband_example_checkpoints/ngc_checkpoints/sc2_ablations/1B_repo_context_Top-level-Depth-first_pp2_64k_64k_2023_10_17_16_35_27/",
#                        dummy_experiment_dir="/toolkit_infiniband_example_checkpoints/ngc_checkpoints/sc2_ablations/dev_1B_repo_context_Random_pp2_64k_64k_2023_10_18_22_20_36/")

