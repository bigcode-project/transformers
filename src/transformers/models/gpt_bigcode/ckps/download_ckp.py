from huggingface_hub import snapshot_download

if __name__ == "__main__":
    snapshot_download("HuggingFaceBR4/starcoder2_7b_4k_smol_data_580000")
    print("done")
