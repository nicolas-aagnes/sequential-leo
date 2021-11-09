import torch

from models.leo import LEO, LEOConfig
from datasets import Dataset2D
import torch
from models.mlp import MLP


def get_model_and_dataloaders(args):
    assert args.model == "leo"
    assert args.dataset == "sine2D"

    config = LEOConfig(
        num_support=args.num_support,
        num_timesteps_pred=args.num_timesteps_pred,
        input_size=2,  # This is for sine 2D.
        encoder_hidden_size=16,
        relation_net_hidden_size=16,
        z_dim=16,
        decoder_hidden_size=32,
        f_theta_hidden_size=128,
    )

    model = LEO(
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lr,
        args.outer_lr,
        args.log_dir,
        config,
    )

    dataset_train = Dataset2D(
        num_support=args.num_support,
        num_query=args.num_query,
        num_timesteps=args.num_timesteps,
        num_timesteps_pred=args.num_timesteps_pred,
        noise=args.noise,
    )
    dataset_val = Dataset2D(
        num_support=args.num_support,
        num_query=args.num_query,
        num_timesteps=args.num_timesteps,
        num_timesteps_pred=args.num_timesteps_pred,
        noise=args.noise,
    )
    dataset_test = Dataset2D(
        num_support=args.num_support,
        num_query=args.num_query,
        num_timesteps=args.num_timesteps,
        num_timesteps_pred=args.num_timesteps_pred,
        noise=args.noise,
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=False,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=False,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=False,
    )

    return model, {
        "train": dataloader_train,
        "val": dataloader_val,
        "test": dataloader_test,
    }
