import torch
import numpy as np
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class BaseMAML:
    """Base class for model agnostic meta-learning models."""

    def __init__(self, num_inner_steps, inner_lr, learn_inner_lr, outer_lr, log_dir):
        self.num_inner_steps = num_inner_steps
        self.inner_lr = inner_lr
        self.learn_inner_lr = learn_inner_lr
        self.outer_lr = outer_lr
        self.log_dir = log_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _inner_loss(self, predictions, target):
        return NotImplementedError

    def _outer_loss(self, predictions, target, parameters):
        raise NotImplementedError

    def _inner_loop(self, x_support, y_support, train):
        raise NotImplementedError

    def _forward(self, x, parameters):
        raise NotImplementedError

    def _outer_step(self, task_batch, train):
        """Computes the outer loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): four tensors with support and query data
                with ground truth labels.
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean loss over the batch, scalar
        """
        outer_loss_batch = []

        for x_support, y_support, x_query, y_query in zip(*task_batch):
            x_support = x_support.to(self.device)
            y_support = y_support.to(self.device)
            x_query = x_query.to(self.device)
            y_query = y_query.to(self.device)

            parameters = self._inner_loop(x_support, y_support, train)

            predictions = self._forward(x_query, parameters)
            outer_loss = self._outer_loss(predictions, y_query, parameters)

            outer_loss_batch.append(outer_loss)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))

        return outer_loss

    def train(self, dataloader_train, dataloader_val, writer):
        """Train.

        Consumes dataloader_train to optimize meta-parameters
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f"Starting training.")
        num_train_tasks, num_val_tasks = 1000, 100
        tot_tasks = 0

        for i_step, task_batch in enumerate(dataloader_train):
            tot_tasks += task_batch[0].shape[0]  # Add number of current batches.

            self.optimizer.zero_grad()
            outer_loss = self._outer_step(task_batch, train=True)
            outer_loss.backward()
            self.optimizer.step()

            print(
                f"Iteration {i_step}, {tot_tasks} tasks: {outer_loss.item():.3f} train loss"
            )
            writer.add_scalar("loss/train", outer_loss.item(), tot_tasks)

            if i_step % 1 == 0:
                val_tasks = 0
                losses = []

                for val_task_batch in dataloader_val:
                    with torch.no_grad():
                        outer_loss = self._outer_step(val_task_batch, train=False)
                        losses.append(outer_loss.item())

                    val_tasks += val_task_batch[0].shape[0]
                    if val_tasks >= num_val_tasks:
                        break

                loss = np.mean(losses)
                print(
                    f"Iteration {i_step}, {tot_tasks} tasks: {loss.item():.3f} val loss"
                )
                writer.add_scalar("loss/val", loss, tot_tasks)

            print()

            if tot_tasks >= num_train_tasks:
                break
