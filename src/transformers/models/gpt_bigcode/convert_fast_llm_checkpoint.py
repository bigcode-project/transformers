import argparse
import os
from pathlib import Path
import re

import torch
from transformers.models.gpt_bigcode.merge_fast_llm_checkpoint import merge_checkpoint
from transformers.models.gpt_bigcode import GPTBigCodeConfig, GPTBigCodeForCausalLM, GPTBigCodeModel


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Path where the converted model is saved"
    )
    args = parser.parse_args(argv)
    # TODO(xrsrke): auto convert checkpoint_dir to Path
    checkpoint_dir = "/admin/home/phuc_nguyen/.cache/huggingface/hub/models--HuggingFaceBR4--starcoder2_7b_4k_smol_data_580000/snapshots/92b6c25cab25f07c367bcc6d773635700a8a287d"
    checkpoint_dir = Path(checkpoint_dir)

    merge_checkpoint(
        checkpoint_dir,
        dummy_experiment_dir=None
    )


if __name__ == "__main__":
    main()
