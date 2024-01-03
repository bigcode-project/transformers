from pathlib import Path
from safetensors import safe_open


if __name__ == "__main__":
    checkpoint_dir = "/admin/home/phuc_nguyen/.cache/huggingface/hub/models--HuggingFaceBR4--starcoder2_7b_4k_smol_data_580000/snapshots/92b6c25cab25f07c367bcc6d773635700a8a287d"
    checkpoint_dir = Path(checkpoint_dir)
    
    assert 1 == 1
