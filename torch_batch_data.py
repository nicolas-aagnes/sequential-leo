import torch
import numpy as np

from batch_data import generate_random_task


class Human36M(torch.utils.data.Dataset):
    """Toy dataset for data following sinusoidal curvers silimar to Finn et al. 2018."""

    def __init__(
        self,
        num_support,
        num_query,
        num_timesteps=50,
        num_timesteps_pred=10,
        data_path="./annotations"
    ):
        self.num_support = num_support
        self.num_query = num_query
        self.num_timesteps = num_timesteps
        self.num_timesteps_pred = num_timesteps_pred
        self.data_path = data_path

    def __getitem__(self, index):
        """Sample a task for the H36M dataset.

        Returns:
            - x_support: Tensor of shape (num_support, num_timesteps, 2)
            - y_support: Tensor of shape (num_support, num_timesteps_pred,    2)
            - x_query:   Tensor of shape (num_query,   num_timesteps, 2)
            - y_query:   Tensor of shape (num_query,   num_timesteps_pred,    2)
            - subject_id, index: Tuple for plotting the ground truth human pose
        """

        train_data, train_label, query_data, query_label = generate_random_task(self.num_timesteps, self.num_timesteps_pred, self.num_support, self.num_query, self.data_path)

        x_support = np.array(train_data)
        y_support = np.array(train_label)
        x_query = np.array(query_data)
        y_query = np.array(query_label)

        return (
            x_support.astype(np.float32),
            y_support.astype(np.float32),
            x_query.astype(np.float32),
            y_query.astype(np.float32),
        )

    def __len__(self):
        return 1000000