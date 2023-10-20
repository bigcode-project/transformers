import re
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
import yaml


def get_checkpoint_paths(experiment_path):
    checkpoints = (Path(experiment_path) / "checkpoints").glob("*")
    # Sort checkpoints by iteration number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.name))
    return [
        [c_name for c_name in checkpoint.glob("*") if re.match(r"\d+", c_name.name)]
        for checkpoint in checkpoints
    ]


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


def merge_checkpoint(experiment_dir, dummy_experiment_dir=None):
    """Load a fast-llm checkpoint and merge the data, tensor, and pipeline-parallel shards"""
    checkpoint_paths = get_checkpoint_paths(experiment_dir)
    config = yaml.safe_load((Path(experiment_dir) / "config.yaml").read_text())

    # Convert the last iteration
    # Load the states from all the ranks
    states = {
        int(c_name.name): torch.load(c_name)
        for c_name in tqdm(checkpoint_paths[-1])
    }
    num_stages = len(states[0]["stages"])
    tensor_parallel = config["tensor_parallel"]
    data_parallel_size = int(config["world_size"] / (tensor_parallel * config["pipeline_parallel"]))

    if dummy_experiment_dir is not None:
        # Use the meta from the dummy checkpoint, and the shard from the actual checkpoint
        dummy_checkpoint_paths = get_checkpoint_paths(dummy_experiment_dir)
        dummy_states = {
            int(c_name.name): torch.load(c_name)
            for c_name in tqdm(dummy_checkpoint_paths[-1])
        }
        for rank, state in dummy_states.items():
            state['shard'] = states[rank]['shard']
        states = dummy_states

    # Gather the data-parallel shards
    # {tp_rank: [[stage_0_shard_0, stage_0_shard_1, ...], [stage_1_shard_0, ...], ...]}
    # {tp_rank: [{fsdp_rank: shard}, ...]}
    fsdp_shards = {
        i: [[None for _ in range(data_parallel_size)] for _ in range(num_stages)]
        for i in range(tensor_parallel)
    }
    
    for rank, state in states.items():
        on_device_stage_shards = extract_stage_shards(state)
        on_device_stage_indices = [i for (i, stage_meta) in enumerate(state["stages"]) if stage_meta["on_device"]]
        for stage_index, stage_shard in zip(on_device_stage_indices, on_device_stage_shards):
            stage_meta = state["stages"][stage_index]
            # fsdp_shards[stage_meta["tp_rank"]][stage_index].append((stage_meta, stage_shard))
            fsdp_shards[stage_meta["tp_rank"]][stage_index][stage_meta["fsdp_rank"]] = stage_shard
    
    # Concatenate the data-parallel shards
    # and get individual weights
    dp_concatenated_shards = {
        tp_rank: [
            extract_individual_weights(
                torch.cat(stage_shards, dim=0),
                states[0]["stages"][stage_index]['content']
            )
            for stage_index, stage_shards in enumerate(fsdp_shards[tp_rank])
        ]
        for tp_rank in range(config["tensor_parallel"])
    }

    # In the tensor-parallel case, concatenate the TP tensors along their TP dimensions.
    tp_concatenated_shards = []
    for stage_index, stage_tp_shards in enumerate(zip(*(dp_concatenated_shards[i] for i in range(tensor_parallel)))):
        stage_content = states[0]["stages"][stage_index]["content"]
        tp_concatenated_shards.append(concatenate_tp_shards(stage_tp_shards, stage_content))

    # In the pipeline-parallel case, merge the stages
    state_dict = {
        weight_meta["name"]: weight
        for stage_meta, stage_weights in zip(states[0]["stages"], tp_concatenated_shards)
        for weight_meta, weight in zip(stage_meta["content"], stage_weights)
    }

    print(f"Total number of parameters: {sum([weight.numel() for weight in state_dict.values()])}")
    return state_dict, config


if __name__ == "__main__":
    merge_checkpoint("/toolkit_infiniband_example_checkpoints/ngc_checkpoints/sc2_ablations/1B_repo_context_Top-level-Depth-first_pp2_64k_64k_2023_10_17_16_35_27/",
                       dummy_experiment_dir="/toolkit_infiniband_example_checkpoints/ngc_checkpoints/sc2_ablations/dev_1B_repo_context_Random_pp2_64k_64k_2023_10_18_22_20_36/")

