import torch
import numpy as np


class Dataset2D(torch.utils.data.Dataset):
    """Toy dataset for data following sinusoidal curvers silimar to Finn et al. 2018."""

    def __init__(self, num_support, num_query, num_frames=50, horizon=10, noise=0.3):
        self.num_support = num_support
        self.num_query = num_query
        self.num_frames = num_frames
        self.horizon = horizon
        self.noise = noise
        self.delta = 0.1

    def __getitem__(self, index):
        """Sample a task for the sinusoidal regression problem.

        Sample a phase and amplitude at random -> This is the definining feature of the task.

        A one-shot learning case would be:
        - sample (num_frames + horizon) x values, ordered by increasing value, and get their correspoinding y values with noise.
        - Xu are the first num_frames points, and Yu are the remaining horizon points.

        Do the above process num_support times for creating x_support and y_support tensors.
        Do the above process num_query   times for creating x_query   and y_query   tensors.

        A task consists of the tuple (x_support, y_support, x_query, y_query)

        The dataset index argument is ignored as we run on args.num_loops * batch_size total tasks.

        Returns:
            - x_support: Tensor of shape (num_support, num_frames, 2)
            - y_support: Tensor of shape (num_support, horizon,    2)
            - x_query:   Tensor of shape (num_query,   num_frames, 2)
            - y_query:   Tensor of shape (num_query,   horizon,    2)
            - amplitude, phase: Tuple for plotting the ground truth sine curve
        """
        amplitude = np.random.uniform(0.1, 5)
        phase = np.random.uniform(0.1, np.pi)

        num_sequences = self.num_support + self.num_query
        num_timesteps = self.num_frames + self.horizon

        x_starts = np.random.uniform(-5, 5 - self.delta * self.horizon, num_sequences)
        x = np.empty((num_sequences, num_timesteps))

        for i, x_start in enumerate(x_starts):  # TODO: This can easily be vecotrized.
            x[i] = np.linspace(
                x_start, x_start + self.delta * num_timesteps, num_timesteps
            )

        y = amplitude * np.sin(x / phase) + self.noise * np.random.randn(*x.shape)

        data = np.concatenate((x[..., None], y[..., None]), axis=2)
        assert data.shape == (num_sequences, num_timesteps, 2), data.shape
        assert (data[:, 1:, 0] > data[:, :-1, 0]).all(), data[0, :, 0]

        x_support = data[: self.num_support, : self.num_frames]
        y_support = data[: self.num_support, self.num_frames :]
        x_query = data[self.num_support :, : self.num_frames]
        y_query = data[self.num_support :, self.num_frames :]

        assert x_support.shape == (
            self.num_support,
            self.num_frames,
            2,
        ), x_support.shape
        assert y_support.shape == (self.num_support, self.horizon, 2), y_support.shape
        assert x_query.shape == (self.num_query, self.num_frames, 2), x_query.shape
        assert y_query.shape == (self.num_query, self.horizon, 2), y_query.shape

        return x_support, y_support, x_query, y_query, (amplitude, phase)

    def __len__(self):
        return 100000000
