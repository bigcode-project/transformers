## Conversion to `transformers`

To convert a model from Megatron-LM to transformers use:
```bash
source ~/.bashrc
export PYTHONPATH=Megatron-LM
export PYTHONPATH=transformers/src:$PYTHONPATH

cd transformers/src/transformers/models

python gpt_bigcode/convert_megatron_checkpoint.py \
    --path_to_checkpoint /fsx/bigcode/experiments/pretraining/starcoder2-1B/checkpoints/iter_0200000/mp_rank_00/model_optim_rng.pt \
    --save_dir /fsx/bigcode/experiments/pretraining/starcoder2-1B/checkpoints/conversions \
    --test_generation \
    --tokenizer_path /fsx/loubna/data/tokenizer/starcoder2-smol-internal-1
```

For `fast-llm` use `convert_fast_llm_checkpoint.py`. For cloning and pushing models from existng iterations directly to HF hub check `push_checkpoints.py`.