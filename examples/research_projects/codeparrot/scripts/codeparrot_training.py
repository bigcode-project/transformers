import os
import time
from argparse import Namespace

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

from accelerate import Accelerator, DistributedType
from arguments import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_scheduler, set_seed
from utils_rholoss import compute_save_control_examples, sanity_check_irred_losses
from utils_training import compute_tflops, evaluate, get_grouped_params, get_lr, setup_logging


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        tokenized=False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.epoch = 0
        self.infinite = infinite
        self.current_size = 0
        self.tokenized = tokenized

        if self.tokenized:
            self.max_buffer_size = seq_length * num_of_sequences
            self.content_field = "input_ids"
        else:
            self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
            self.content_field = "content"

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            if self.tokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield torch.tensor(input_ids)

    def shuffle(self, buffer_size=1000):
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)


def create_dataloaders(args):
    train_data = load_dataset(args.dataset_name_train, split="train", use_auth_token=True)
    train_data = train_data.shuffle(seed=args.seed)
    if args.dataset_name_train == args.dataset_name_valid:
        # Split the dataset into train and validation
        data = train_data.train_test_split(test_size=0.005, shuffle=False, seed=args.seed)
        train_data = data["train"]
        valid_data = data["test"]
        print(f"Size of train set: {len(train_data)} and validation set {len(valid_data)}")
    else:
        valid_data = load_dataset(args.dataset_name_valid, split="train", use_auth_token=True)
    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=args.seq_length, tokenized=args.tokenized
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=args.seq_length, tokenized=args.tokenized
    )
    train_dataset = train_dataset.shuffle(buffer_size=args.shuffle_buffer)
    # ShufflerIterDataPipe does not work in torch > 1.11
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader


def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        accelerator.log(metrics, step)


# Settings
parser = HfArgumentParser(TrainingArguments)
args = parser.parse_args()

# Accelerator
accelerator = Accelerator(log_with=["wandb", "tensorboard"], logging_dir=f"{args.save_dir}/log")
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

args = Namespace(**vars(args), **acc_state)
samples_per_step = accelerator.state.num_processes * args.train_batch_size
set_seed(args.seed)

# Logging
logger, run_name = setup_logging(accelerator, args)
logger.info(accelerator.state)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(args)

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
)
accelerator.register_for_checkpointing(lr_scheduler)


# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Train model
model.train()
completed_steps = 0
t_start = time.time()
loss_tracking = 0

samples_per_step = args.train_batch_size * accelerator.state.num_processes
total_samples = args.max_train_steps * samples_per_step

if args.compute_irred_losses:
    irred_losses = torch.zeros(total_samples, device="cpu")
    compute_save_control_examples(
        accelerator,
        train_dataloader,
        args.train_batch_size,
        limit_control_examples=600,
        seq_length=args.seq_length,
        save_dir="./control_examples",
    )
    if accelerator.is_main_process:
        print(f"Total number of samples: {total_samples}")
        print("Irreducible loss examples were saved in ./control_examples for future sanity checks on the order")

elif args.irred_losses and not args.compute_irred_losses:
    # Should be of shape total_samples
    irred_losses = torch.load(args.irred_losses, map_location=torch.device("cpu"))
    assert irred_losses.shape[0] == total_samples, print(
        f"shape of irred_losses: {irred_losses.shape[0]}, len of train_dataloader: {total_samples}"
    )

if args.selection_method and args.selection_method == "rholoss":
    if accelerator.is_main_process:
        print("Running sanity checks for irreducible losses")
    # run sanity checks to verify the order of irreducible losses wrt current batches
    sanity_check_irred_losses(
        accelerator,
        train_dataloader,
        args.train_batch_size,
        limit_control_examples=600,
        seq_length=args.seq_length,
        save_dir="./control_examples",
    )

for step, batch in enumerate(train_dataloader, start=1):
    if args.selection_method:
        # select a smaller batch of examples to train on using a selection method
        if args.selection_method == "uniform":
            batch = batch[torch.randperm(batch.size(0))[: args.train_batch_size_select]]
        elif args.selection_method == "rholoss":
            # in this setting global_batch_size = train_batch_size x nb_workers
            with torch.no_grad():
                losses = []
                # To avoid running OOM, compute the loss on a smaller batches 
                sub_batches = torch.split(batch, args.gradient_accumulation_steps)
                for sub_batch in sub_batches:
                    # we use reduction="none" in GPT2LMHeadModel loss implementation (install transformers from a fork)
                    loss = model(sub_batch, labels=sub_batch, use_cache=False).loss
                    loss = loss.view(sub_batch.size(0), -1).mean(dim=1)
                    losses.append(loss)
                losses = torch.cat(losses, dim=0)
                assert loss.shape == torch.Size(
                    [args.train_batch_size]
                ), "make sure you are using GPT2LMHeadModel with reduction=none in the loss"
                # TODO check data at the end that may be duplicated to divide batch equally among all workers
                losses = accelerator.gather(loss)
                cur_irred_losses = irred_losses[(step - 1) * samples_per_step : step * samples_per_step]
                try:
                    losses.shape == cur_irred_losses.shape
                except:
                    print(
                        f"Size mismatch between training losses {losses.shape} and irreducible losses {cur_irred_losses.shape}"
                    )
                red_losses = losses - cur_irred_losses.to(losses.device)
                print(f"shape red_losses {red_losses.shape} and max is {torch.max(red_losses[0])}")
                # Select the top args.train_batch_size_select losses & produce a new batch
                top_losses, top_indices = torch.topk(red_losses, args.train_batch_size_select)
                batches = accelerator.gather(batch)
                batch = torch.index_select(batches, 0, top_indices)
                # assert first element has the highest loss
                assert torch.eq(top_losses[0], torch.max(red_losses)), "first element should have the highest loss"
                print(f"new size of batch is {batch.shape}")

    if args.compute_irred_losses:
        # compute irreducible losses over the entire dataset and exit
        with torch.no_grad():
            # we use reduction="none" in GPT2LMHeadModel loss implementation
            loss = model(batch, labels=batch, use_cache=False).loss
            loss = loss.view(batch.size(0), -1).mean(dim=1)
            assert loss.shape == torch.Size(
                [args.train_batch_size]
            ), "make sure you are using GPT2LMHeadModel with reduction=none in the loss"
            losses = accelerator.gather(loss)
            try:
                irred_losses[(step - 1) * samples_per_step : step * samples_per_step] = losses
            except:
                print(
                    f"Size mismatch, step {step} between current losses {losses.shape} and irreducible losses \
                {irred_losses[(step-1) * samples_per_step: step * samples_per_step].shape}"
                )
                break
        if step >= args.max_train_steps:
            break
        continue

    # model training
    # TODO! 32 batch doesn't fit => split batch and add "proper" grad accumulation
    # we are using reduction="none" in GPT2LMHeadModel loss => we add .mean() over tokens
    loss = model(batch, labels=batch, use_cache=False).loss.mean()
    # no need to do accelerate gather we would just be repeating the loss (workers have the same batch)
    loss_tracking += loss.item() / args.gradient_accumulation_steps
    log_metrics(step, {"samples": step * samples_per_step, "loss_per_step/train": loss.item()})
    loss = loss / args.gradient_accumulation_steps
    if step % args.gradient_accumulation_steps != 0:
        # Prevent backward from doing gradient all_reduce in every step
        if accelerator.distributed_type == DistributedType.MULTI_GPU:
            with model.no_sync():
                accelerator.backward(loss)
        else:
            accelerator.backward(loss)
    else:
        lr = get_lr(optimizer)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        elapsed_time = time.time() - t_start
        tflops = compute_tflops(model, tokenizer, elapsed_time, accelerator, args)
        log_metrics(
            step,
            {
                "steps": completed_steps,
                "loss/train": loss_tracking,
                "lr": lr,
                "tflops": tflops,
                "time_per_iteration": elapsed_time,
            },
        )
        t_start = time.time()
        loss_tracking = 0
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info("Evaluating and saving model checkpoint")
        eval_loss, perplexity = evaluate(model, accelerator, eval_dataloader, args)
        log_metrics(step, {"loss/eval": eval_loss, "perplexity": perplexity})
        accelerator.wait_for_everyone()
        save_dir = os.path.join(args.save_dir, f"step_{step}")
        accelerator.save_state(save_dir)
        model.train()
    if completed_steps >= args.max_train_steps:
        break

# Save irred losses
if args.compute_irred_losses:
    if accelerator.is_main_process:
        print(f"saving irred losses with shape: {irred_losses.shape}")
        torch.save(irred_losses, "irred_losses.pt")
    exit()

# Evaluate and save the last checkpoint
logger.info("Evaluating and saving model after training")
eval_loss, perplexity = evaluate(model, accelerator, eval_dataloader, args)
log_metrics(step, {"loss/eval": eval_loss, "perplexity": perplexity})
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(args.save_dir, save_function=accelerator.save)
save_dir = os.path.join(args.save_dir, f"step_{step}")
accelerator.save_state(save_dir)
