import os

import torch


def get_control_examples(accelerator, dataloader, train_batch_size, limit_control_examples=600, seq_length=1024):
    """get control examples for computing the irreducible losses over the entire dataset"""
    control_examples = torch.zeros((limit_control_examples, seq_length), device="cpu").type(torch.LongTensor)
    for step, batch in enumerate(dataloader, start=1):
        indexes = list(
            range(
                (step - 1) * train_batch_size * accelerator.state.num_processes,
                step * train_batch_size * accelerator.state.num_processes,
            )
        )
        if max(indexes) >= limit_control_examples:
            break
        with torch.no_grad():
            batches = accelerator.gather(batch)
            control_examples[indexes] = batches.cpu()
    return control_examples


def compute_save_control_examples(
    accelerator,
    dataloader,
    train_batch_size,
    limit_control_examples=600,
    seq_length=1024,
    save_dir="./control_examples",
):
    """compute and save control examples for computing the irreducible losses over the entire dataset"""
    control_examples = get_control_examples(
        accelerator, dataloader, train_batch_size, limit_control_examples, seq_length
    )
    if accelerator.is_main_process:
        # save each losses and examples
        os.makedirs(save_dir, exist_ok=True)
        torch.save(control_examples, f"{save_dir}/control_examples.pt")


def sanity_check_irred_losses(
    accelerator,
    dataloader,
    train_batch_size,
    limit_control_examples=600,
    seq_length=1024,
    save_dir="./control_examples",
):
    """sanity check of the order of loaded irreducible losses wrt batches of the current dataset"""
    loaded_examples = torch.load(f"{save_dir}/control_examples.pt")
    control_examples = get_control_examples(
        accelerator, dataloader, train_batch_size, limit_control_examples, seq_length
    )
    # check if the loaded tensors are the same as the ones we saved
    assert torch.all(torch.eq(loaded_examples, control_examples))
    if accelerator.is_main_process:
        print("Sanity check for irreducible loss order passed")
