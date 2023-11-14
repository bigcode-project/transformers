import os
import argparse
import re
import subprocess
from pathlib import Path

from huggingface_hub import Repository

from transformers.models.gpt_bigcode.convert_fast_llm_checkpoint import main as convert


"""
Script to upload Fast-llm checkpoints to a HF repo on the Hub. The script clones/creates a repo on the Hub, checks out
a branch `--branch_name`, and converts each `iter_` checkpoint and saves it as a commit on that branch.
"""


def get_iter_number(iter_dir: str):
    m = re.match(r"(\d+)", iter_dir)
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
    parser.add_argument(
        "--iters",
        type=int,
        nargs='+',
        default=None,
        help="Specify a list of iterations to push. If None (default), will potentially push all the checkpoints (subject to iter_interval)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer file to commit before the checkoints.",
    )
    parser.add_argument(
        "--push_past_iters",
        action="store_true",
        default=False,
        help="If True, also push iterations that are lower than the last commit.",
    )
    args, argv = parser.parse_known_args(argv)

    save_dir = args.save_dir or args.exp_dir / "hf_checkpoints"

    hf_repo = Repository(save_dir, clone_from=args.repo_name)
    hf_repo.git_checkout(args.branch_name, create_branch_ok=True)
    
    # Pull latest changes
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    git_pull_output = subprocess.run(["git", "pull"], cwd=save_dir, capture_output=True, env=env)
    print(git_pull_output)
    
    # Find last checkpoint that was uploaded
    head_hash = hf_repo.git_head_hash()
    commit_msg = subprocess.check_output(["git", "show", "-s", "--format=%B", head_hash], cwd=save_dir).decode()
    try:
        last_commit_iter = get_iter_number(commit_msg.strip())
        print(f"Last commit iteration: {last_commit_iter}")
    except ValueError:
        last_commit_iter = -1

    # The checkpoint dirs should be in ascending iteration order, so that the last commit corresponds to the latest checkpoint
    ckpt_dirs = [x for x in (args.exp_dir / "checkpoints").iterdir() if re.match(r"(\d+)", x.name) and x.is_dir()]
    if args.iters is not None:
        args.iters = [int(n) for n in args.iters]
        ckpt_dirs = [p for p in ckpt_dirs if get_iter_number(p.name) in args.iters]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda p: get_iter_number(p.name))
    print(f"Found the following checkpoints: {ckpt_dirs}")
    
    if args.tokenizer is not None:
        raise NotImplementedError("Push tokenizer not implemented yet")

    for ckpt_dir in ckpt_dirs:
        iter_number = get_iter_number(ckpt_dir.name)
        if not args.push_past_iters and iter_number <= last_commit_iter:
            continue
        if iter_number % args.iter_interval == 0:
            print(f"Converting iteration {iter_number}")
            convert(argv + [f"--save_dir={str(save_dir)}", f"--checkpoint_dir={ckpt_dir}"])
            print(f"Pushing iteration {iter_number}")
            hf_repo.push_to_hub(commit_message=f"{ckpt_dir.name}")


if __name__ == "__main__":
    main()