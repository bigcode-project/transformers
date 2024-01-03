import argparse
import os
from pathlib import Path

import torch
from transformers.models.gpt_bigcode.merge_fast_llm_checkpoint import merge_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        # default="/admin/home/phuc_nguyen/.cache/huggingface/hub/models--HuggingFaceBR4--starcoder2_7b_4k_smol_data_580000/snapshots/92b6c25cab25f07c367bcc6d773635700a8a287d",
        help="Path where the converted model is saved"
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        # default="./",
        help="Path where the converted model is saved"
    )
    args = parser.parse_args(argv)
    
    print("start")

    # TODO(xrsrke): auto convert checkpoint_dir to Path
    # checkpoint_dir = "/admin/home/phuc_nguyen/.cache/huggingface/hub/models--HuggingFaceBR4--starcoder2_7b_4k_smol_data_580000/snapshots/92b6c25cab25f07c367bcc6d773635700a8a287d"
    # checkpoint_dir = Path(checkpoint_dir)

    state_dict = merge_checkpoint(args.checkpoint_dir)
    
    save_dir = args.save_dir or args.checkpoint_dir / "converted"
    output_checkpoint_file = os.path.join(save_dir, "pytorch_model.bin")
    
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(state_dict, output_checkpoint_file)
    print(f'Done!')
    
    # # Compare
    # def compare_state_dicts(dict1, dict2):
    #     # Compare keys
    #     if set(dict1.keys()) != set(dict2.keys()):
    #         return "Different keys"

    #     # Compare shapes and values
    #     for key in dict1:
    #         if dict1[key].shape != dict2[key].shape:
    #             return f"Different shape for key: {key}"
    #         if not torch.allclose(dict1[key], dict2[key]):
    #             return f"Different values for key: {key}"

    #     return "State dictionaries are identical"

    # ref_state_dict = torch.load("/fsx/phuc/projects/starcoder/transformers-starcoder/src/transformers/models/gpt_bigcode/merged_checkpoint.pth")
    # result = compare_state_dicts(state_dict, ref_state_dict)
    # print(result)



if __name__ == "__main__":
    main()
