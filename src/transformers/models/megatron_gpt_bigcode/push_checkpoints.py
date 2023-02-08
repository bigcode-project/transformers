import argparse
import re
import subprocess
from pathlib import Path

from huggingface_hub import Repository

from .convert_megatron_gpt_bigcode_checkpoint import main as convert


"""
Script to upload Megatron checkpoints to a HF repo on the Hub.
The script clones/creates a repo on the Hub, checks out a branch `--branch_name`,
and converts each `iter_` checkpoint and saves it as a commit on that branch.
"""


def get_iter_number(iter_dir: str):
    m = re.match(r"iter_(\d+)", iter_dir)
    if m is not None:
        return int(m.group(1))
    else:
        raise ValueError(f"Invalid directory name: {iter_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=Path, required=True, help="Path to experiment folder.")
    parser.add_argument("--repo_name", required=True, help="Name of repository on the Hub in 'ORG/NAME' format.")
    parser.add_argument("--branch_name", required=True, help="Name of branch in repository to save experiments.")
    parser.add_argument(
        "--save_dir",
        type=Path,
        help="Path where repository is cloned to locally. Will use {exp_dir}/hf_checkpoints if not provided",
    )
    parser.add_argument(
        "--iter_interval",
        type=int,
        default=1,
        help="Iteration number must be divisble by iter_interval in order to be pushed",
    )
    args, argv = parser.parse_known_args(argv)

    save_dir = args.save_dir or args.exp_dir / "hf_checkpoints"

    hf_repo = Repository(save_dir, clone_from=args.repo_name)
    hf_repo.git_checkout(args.branch_name, create_branch_ok=True)
    # Find last checkpoint that was uploaded
    head_hash = hf_repo.git_head_hash()
    commit_msg = subprocess.check_output(["git", "show", "-s", "--format=%B", head_hash], cwd=save_dir).decode()
    try:
        last_commit_iter = get_iter_number(commit_msg.strip())
        print(f"Last commit iteration: {last_commit_iter}")
    except ValueError:
        last_commit_iter = -1

    # The checkpoint dirs should be in ascending iteration order, so that the last commit corresponds to the latest checkpoint
    ckpt_dirs = sorted([x for x in args.exp_dir.iterdir() if x.name.startswith("iter_") and x.is_dir()])

    for ckpt_dir in ckpt_dirs:
        iter_number = get_iter_number(ckpt_dir.name)
        if iter_number <= last_commit_iter:
            continue
        if iter_number % args.iter_interval == 0:
            print(f"Converting iteration {iter_number}")
            # TODO: this only works for 1-way tensor/pipeline parallelism
            file_path = next((ckpt_dir / "mp_rank_00").glob("*.pt"))
            convert(argv + [f"--save_dir={str(save_dir)}", str(file_path)])
            print(f"Pushing iteration {iter_number}")
            hf_repo.push_to_hub(commit_message=f"{ckpt_dir.name}")


if __name__ == "__main__":
    main()
