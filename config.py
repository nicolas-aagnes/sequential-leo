import torch

from models.leo import LEO, LEOConfig
from datasets import Dataset2D, Human36M
import torch
from models.mlp import MLP
from models.maml_new import MAML, MAMLConfig
import copy


def get_model_and_dataloaders(args):
    if args.dataset == "sine2D":
        dataset_train = Dataset2D(
            num_support=args.num_support,
            num_query=args.num_query,
            num_timesteps=args.num_timesteps,
            num_timesteps_pred=args.num_timesteps_pred,
            noise=args.sine2d_noise,
            delta=args.sine2d_delta,
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        )
        dataloader_val = copy.deepcopy(dataloader_train)
        dataloader_test = copy.deepcopy(dataloader_train)
        input_size = 2

    elif args.dataset == "human36":
        dataset_train = Human36M(
            num_support=args.num_support,
            num_query=args.num_query,
            num_timesteps=args.num_timesteps,
            num_timesteps_pred=args.num_timesteps_pred,
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        )
        dataloader_val = copy.deepcopy(dataloader_train)
        dataloader_test = copy.deepcopy(dataloader_train)
        input_size = 51
    else:
        raise NotImplementedError

    if args.model == "leo":
        config = LEOConfig(
            num_support=args.num_support,
            num_timesteps_pred=args.num_timesteps_pred,
            input_size=input_size,  # This is for sine 2D.
            encoder_hidden_size=args.leo_encoder_hidden_size,
            relation_net_hidden_size=args.leo_relation_net_hidden_size,
            z_dim=args.leo_z_dim,
            decoder_hidden_size=args.leo_decoder_hidden_size,
            f_theta_hidden_size=args.leo_f_theta_hidden_size,
        )

        model = LEO(
            args.num_inner_steps,
            args.inner_lr,
            args.learn_inner_lr,
            args.outer_lr,
            args.log_dir,
            config,
        )
    elif args.model == "maml":
        config = MAMLConfig(
            num_timesteps_pred=args.num_timesteps_pred,
            input_size=input_size,
            hidden_size=args.maml_hidden_size,
        )
        model = MAML(
            args.num_inner_steps,
            args.inner_lr,
            args.learn_inner_lr,
            args.outer_lr,
            args.log_dir,
            config,
        )
    else:
        raise NotImplementedError

    return model, {
        "train": dataloader_train,
        "val": dataloader_val,
        "test": dataloader_test,
    }
