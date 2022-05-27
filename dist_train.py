#!/usr/bin/python
import torch
import torch.nn as nn
import ml_collections
import yaml
import lib.utils.bookkeeping as bookkeeping
from pathlib import Path
import torch.utils.tensorboard as tensorboard
from tqdm import tqdm
import sys
import signal
import os
import argparse


import lib.models.models as models
import lib.models.model_utils as model_utils
import lib.datasets.datasets as datasets
import lib.datasets.dataset_utils as dataset_utils
import lib.losses.losses as losses
import lib.losses.losses_utils as losses_utils
import lib.training.training as training
import lib.training.training_utils as training_utils
import lib.optimizers.optimizers as optimizers
import lib.optimizers.optimizers_utils as optimizers_utils
import lib.loggers.loggers as loggers
import lib.loggers.logger_utils as logger_utils

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, unique_num):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355 + unique_num)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, cfg, unique_num, custom_name=None):
    # unique_num is a number that differentiates runs in a batch that may
    # run on the same machine and we don't want them using the same port
    # or ddp_store file
    # ideally in the range 0 - 10

    print("Training with config", cfg.experiment_name)
    print(f"Rank: {rank}/{world_size}")


    setup(rank, world_size, unique_num)
    ddp_store = dist.FileStore(f"/tmp/tldr_ddpstore_{unique_num}")


    preempted_path = Path("null")
    if rank == 0:
        if cfg.saving.enable_preemption_recovery:

            preempted_path = bookkeeping.check_for_preempted_run(
                cfg.save_location,
                cfg.saving.preemption_start_day_YYYYhyphenMMhyphenDD,
                cfg,
                cfg.saving.prepare_to_resume_after_timeout
            )

        if preempted_path.as_posix() == "null":
            save_dir, checkpoint_dir, config_dir = \
                bookkeeping.create_experiment_folder(
                    cfg.save_location,
                    cfg.experiment_name if custom_name is None else custom_name,
                    custom_name is None
            )
            bookkeeping.save_config_as_yaml(cfg, config_dir)
            bookkeeping.save_git_hash(save_dir)
        else:
            print("Resuming from preempted run: ", preempted_path)
            save_dir = preempted_path
            checkpoint_dir, config_dir = bookkeeping.create_inner_experiment_folders(save_dir)

        ddp_store.set("preempted_path", preempted_path.as_posix())
        ddp_store.set("save_dir", save_dir.as_posix())
        ddp_store.set("checkpoint_dir", checkpoint_dir.as_posix())
        ddp_store.set("config_dir", config_dir.as_posix())

    dist.barrier() # wait for rank 0 process to get/create all the folders
    preempted_path = Path(ddp_store.get("preempted_path").decode("utf-8"))
    save_dir = Path(ddp_store.get("save_dir").decode("utf-8"))
    checkpoint_dir = Path(ddp_store.get("checkpoint_dir").decode("utf-8"))
    config_dir = Path(ddp_store.get("config_dir").decode("utf-8"))

    # writer is DummyWriter for non zero ranks
    writer = bookkeeping.setup_tensorboard(save_dir, rank)

    if cfg.device == "cpu":
        device = torch.device("cpu")
    elif cfg.device == "cuda":
        device = torch.device(f"cuda:{rank}")

    model = model_utils.create_model(cfg, device, rank)
    print("number of parameters: ", sum([p.numel() for p in model.parameters()]))

    dataset = dataset_utils.get_dataset(cfg, device)
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
        shuffle=cfg.data.shuffle)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=cfg.data.batch_size//world_size,
        sampler=dist_sampler)

    loss = losses_utils.get_loss(cfg)

    training_step = training_utils.get_train_step(cfg)

    optimizer = optimizers_utils.get_optimizer(model.parameters(), cfg)

    state = {
        'model': model,
        'optimizer': optimizer,
        'n_iter': 0
    }

    if rank == 0:
        bookkeeping.setup_preemption(
            save_dir, checkpoint_dir, state,
            cfg.saving.num_checkpoints_to_keep,
            cfg.saving.prepare_to_resume_after_timeout
        )

    dist.barrier()


    mapping = {'cuda:0': 'cuda:%d' % rank}
    if not preempted_path.as_posix() == "null":
        print("loading state from premption: ", preempted_path)
        state = bookkeeping.resume_training(preempted_path, state, mapping)
    elif cfg.init_model_path is not None:
        print("loading state from init model: ", cfg.init_model_path)
        loaded_state = torch.load(cfg.init_model_path, map_location=mapping)
        state['model'].load_state_dict(loaded_state['model'])
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['n_iter'] = loaded_state['n_iter']


    low_freq_loggers = []
    for logger in cfg.saving.low_freq_loggers:
        low_freq_loggers.append(logger_utils.get_logger(logger))

    exit_flag = False

    while True:
        for minibatch in tqdm(dataloader):

            training_step.step(state, minibatch, loss, writer)

            if rank == 0:
                if state['n_iter'] % cfg.saving.checkpoint_freq == 0 or state['n_iter'] == cfg.training.n_iters-1:
                    bookkeeping.save_checkpoint(checkpoint_dir, state,
                        cfg.saving.num_checkpoints_to_keep)

                if state['n_iter'] % cfg.saving.checkpoint_archive_freq == 0:
                    bookkeeping.save_archive_checkpoint(checkpoint_dir, state)

                if state['n_iter'] % cfg.saving.log_low_freq == 0 or state['n_iter'] == cfg.training.n_iters-1:
                    for logger in low_freq_loggers:
                        logger(state=state, cfg=cfg, writer=writer,
                            minibatch=minibatch, dataset=dataset)


            state['n_iter'] += 1
            if state['n_iter'] > cfg.training.n_iters - 1:
                exit_flag = True
                break

        if exit_flag:
            break

    if rank == 0:
        bookkeeping.no_more_preemption_recovery_needed(save_dir)
        writer.close()

    return save_dir



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args, unknown_args = parser.parse_known_args()
    if args.config == 'cifar10':
        from config.train.cifar10_distributed import get_config
    else:
        raise NotImplementedError


    cfg = get_config()
    world_size = cfg.num_gpus
    mp.spawn(main,
        args=(world_size, cfg, 0),
        nprocs=world_size,
        join=True
    )
