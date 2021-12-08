"""Example launcher for a hyperparameter search on SLURM.

This example shows how to use gpus on SLURM with PyTorch.
"""
import torch

from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster


from argparse import ArgumentParser
from typing import Any, Dict


import torch.utils.tensorboard as tensorboard
import argparse
from config import get_model_and_dataloaders
from models.maml_new import MAML, MAMLConfig

import numpy as np
import torch

import torch.utils.tensorboard as tensorboard
import argparse
from config import get_model_and_dataloaders
from models.leo import LEO, LEOConfig
import sys


def train(args, cluster):
    print(args)
    sys.stdout.flush()
    from pathlib import Path

    path = Path(args.test_tube_slurm_cmd_path)
    trial_folder, trial_name = path.parent.parent, path.name.strip(".sh")
    writer = tensorboard.SummaryWriter(log_dir=str(trial_folder / "logs" / trial_name))
    model, dataloaders = get_model_and_dataloaders(args)

    for i, results in enumerate(
        model.train(dataloaders["train"], dataloaders["val"], writer, args, True)
    ):
        sys.stdout.flush()

    print("Done training. Starting testing...")
    sys.stdout.flush()
    test_loss = model.eval(dataloaders["test"], args.num_test_tasks)
    print("test_loss", test_loss)
    writer.add_scalar("loss/test", test_loss)


if __name__ == "__main__":
    parser = HyperOptArgumentParser(strategy="random_search")

    # SLURM arguments.
    parser.add_argument("--test_tube_exp_name", default="leo")

    # Dataset arguments.
    parser.add_argument("--dataset", default="human36", choices=["sine2D", "human36"])
    parser.add_argument(
        "--num_support",
        default=1,
        type=int,
        help="number of sequences per task (equivalent to K in K-shot learning problem)",
    )
    parser.add_argument(
        "--num_query", default=15, type=int, help="number of sequences in query task"
    )
    parser.add_argument(
        "--num_timesteps",
        default=10,
        type=int,
        help="number of datapoints in the given sequences with ground truth labels",
    )
    parser.add_argument(
        "--num_timesteps_pred",
        default=4,
        type=int,
        help="number of timesteps to predict",
    )
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--num_train_tasks", default=100000, type=int)
    parser.add_argument("--num_val_tasks", default=512, type=int)
    parser.add_argument("--num_test_tasks", default=10000, type=int)

    # Sine 2D specific arguments.
    parser.add_argument("--sine2d_noise", default=0.0, type=float)
    parser.add_argument("--sine2d_delta", default=0.3, type=float)

    # Model arguments.
    parser.add_argument("--model", default="leo", choices=["leo", "maml"])
    parser.opt_list(
        "--inner_lr",
        default=0.4,
        options=[0.3, 0.1, 0.03, 0.01],
        type=float,
        tunable=True,
    )
    parser.add_argument("--learn_inner_lr", action="store_true")
    parser.opt_list(
        "--outer_lr",
        default=0.001,
        options=[0.001, 0.0003, 0.0001],
        type=float,
        tunable=True,
    )
    parser.opt_list(
        "--num_inner_steps",
        default=1,
        options=[1, 2, 3],
        type=int,
        tunable=True,
    )

    # LEO specific arguments.
    parser.opt_list(
        "--leo_encoder_hidden_size",
        default=512,
        options=[128, 256, 512],
        type=int,
        tunable=True,
    )
    parser.opt_list(
        "--leo_relation_net_hidden_size",
        default=512,
        options=[128, 256, 512],
        type=int,
        tunable=True,
    )
    parser.opt_list(
        "--leo_z_dim",
        default=512,
        options=[32, 64, 128],
        type=int,
        tunable=True,
    )
    parser.opt_list(
        "--leo_decoder_hidden_size",
        default=512,
        options=[128, 256, 512],
        type=int,
        tunable=True,
    )
    parser.opt_list(
        "--leo_f_theta_hidden_size",
        default=512,
        options=[128, 256, 512],
        type=int,
        tunable=True,
    )

    # MAML Specific parameters.
    parser.opt_list(
        "--maml_hidden_size",
        default=512,
        options=[128, 256, 512],
        type=int,
        tunable=True,
    )

    args = parser.parse_args()
    args.log_dir = f"test/{args.dataset}"

    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=args,
        log_path=args.log_dir,
        python_cmd="python",
    )

    # Add commands to the non-SLURM portion.
    cluster.add_command("cd /vision/u/naagnes/github/sequential-leo")
    cluster.add_command("source .svl/bin/activate")

    # SLURM commands.
    cluster.add_slurm_cmd(cmd="partition", value="svl", comment="")
    cluster.add_slurm_cmd(cmd="qos", value="normal", comment="")
    cluster.add_slurm_cmd(cmd="time", value="4:00:00", comment="")
    cluster.add_slurm_cmd(cmd="ntasks-per-node", value=1, comment="")
    cluster.add_slurm_cmd(cmd="cpus-per-task", value=16, comment="")
    cluster.add_slurm_cmd(cmd="mem", value="30G", comment="")

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.gpu_type = "titanrtx"

    # Run experiments.
    cluster.optimize_parallel_cluster_gpu(train, nb_trials=8, job_name=args.model)
