import torch.utils.tensorboard as tensorboard
import argparse
from config import get_model_and_dataloaders


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f"./logs/"
    print(f"log_dir: {log_dir}")
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    model, dataloaders = get_model_and_dataloaders(args)

    if args.checkpoint_step > -1:
        raise NotImplementedError
        model.load(args.checkpoint_step)
    else:
        print("Checkpoint loading skipped.")

    if not args.test:
        model.train(dataloaders["train"], dataloaders["val"], writer)
    else:
        raise NotImplementedError
        model.test(dataloaders["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a MAML variant!")
    parser.add_argument("--model", choices=["leo"], default="leo", help="model name")
    parser.add_argument(
        "--dataset", choices=["sine2D"], default="sine2D", help="dataset name"
    )
    parser.add_argument(
        "--num_support",
        type=int,
        default=1,
        help="number of support clips, equal to K in K-shot learning problem",
    )
    parser.add_argument(
        "--num_query",
        type=int,
        default=15,
        help="number of query examples",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=10,
        help="number of frames with ground truth labels",
    )
    parser.add_argument(
        "--num_timesteps_pred",
        type=int,
        default=4,
        help="number of timesteps for which model has to output a prediction",
    )
    parser.add_argument(
        "--num_inner_steps", type=int, default=1, help="number of inner-loop updates"
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=0.4,
        help="inner-loop learning rate initialization",
    )
    parser.add_argument(
        "--learn_inner_lr",
        default=False,
        action="store_true",
        help="whether to optimize inner-loop learning rate",
    )
    parser.add_argument(
        "--outer_lr", type=float, default=0.001, help="outer-loop learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="number of tasks per outer-loop update",
    )
    parser.add_argument(
        "--num_train_tasks",
        type=int,
        default=10000,
        help="number of total tasks to train on",
    )
    parser.add_argument(
        "--test", default=False, action="store_true", help="train or test"
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=-1,
        help=(
            "checkpoint iteration to load for resuming "
            "training, or for evaluation (-1 is ignored)"
        ),
    )
    parser.add_argument(
        "--log_dir", type=str, default=None, help="directory to save to or load from"
    )

    main_args = parser.parse_args()
    main(main_args)
