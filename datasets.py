import torch
import numpy as np


class Dataset2D(torch.utils.data.Dataset):
    """Toy dataset for data following sinusoidal curvers silimar to Finn et al. 2018."""

    def __init__(
        self,
        num_support,
        num_query,
        num_timesteps=10,
        num_timesteps_pred=4,
        noise=0.3,
        delta=0.1,
        return_amplitude_and_phase=False,
    ):
        self.num_support = num_support
        self.num_query = num_query
        self.num_timesteps = num_timesteps
        self.num_timesteps_pred = num_timesteps_pred
        self.noise = noise
        self.delta = delta
        self.return_amplitude_and_phase = return_amplitude_and_phase

    def __getitem__(self, index):
        """Sample a task for the sinusoidal regression problem.

        Sample a phase and amplitude at random -> This is the definining feature of the task.

        A one-shot learning case would be:
        - sample (num_timesteps + num_timesteps_pred) x values, ordered by increasing value, and get their correspoinding y values with noise.
        - Xu are the first num_timesteps points, and Yu are the remaining num_timesteps_pred points.

        Do the above process num_support times for creating x_support and y_support tensors.
        Do the above process num_query   times for creating x_query   and y_query   tensors.

        A task consists of the tuple (x_support, y_support, x_query, y_query)

        The dataset index argument is ignored as the number of tasks trained on is determined outside this class.

        Returns:
            - x_support: Tensor of shape (num_support, num_timesteps, 2)
            - y_support: Tensor of shape (num_support, num_timesteps_pred,    2)
            - x_query:   Tensor of shape (num_query,   num_timesteps, 2)
            - y_query:   Tensor of shape (num_query,   num_timesteps_pred,    2)
            - amplitude, phase: Tuple for plotting the ground truth sine curve
        """
        amplitude = np.random.uniform(0.1, 5)
        phase = np.random.uniform(0.3, np.pi)

        num_sequences = self.num_support + self.num_query
        num_timesteps = self.num_timesteps + self.num_timesteps_pred

        x_starts = np.random.uniform(-5, 5 - self.delta * num_timesteps, num_sequences)
        x = np.empty((num_sequences, num_timesteps))

        for i, x_start in enumerate(x_starts):  # TODO: This can easily be vecotrized.
            x[i] = np.linspace(
                x_start, x_start + self.delta * num_timesteps, num_timesteps
            )

        y = amplitude * np.sin(x / phase) + self.noise * np.random.randn(*x.shape)

        data = np.concatenate((x[..., None], y[..., None]), axis=2)
        assert data.shape == (num_sequences, num_timesteps, 2), data.shape
        assert (data[:, 1:, 0] > data[:, :-1, 0]).all(), data[0, :, 0]

        x_support = data[: self.num_support, : self.num_timesteps]
        y_support = data[: self.num_support, self.num_timesteps :]
        x_query = data[self.num_support :, : self.num_timesteps]
        y_query = data[self.num_support :, self.num_timesteps :]

        assert x_support.shape == (
            self.num_support,
            self.num_timesteps,
            2,
        ), x_support.shape
        assert y_support.shape == (
            self.num_support,
            self.num_timesteps_pred,
            2,
        ), y_support.shape
        assert x_query.shape == (self.num_query, self.num_timesteps, 2), x_query.shape
        assert y_query.shape == (
            self.num_query,
            self.num_timesteps_pred,
            2,
        ), y_query.shape

        if self.return_amplitude_and_phase:
            return x_support, y_support, x_query, y_query, (amplitude, phase)

        return (
            x_support.astype(np.float32),
            y_support.astype(np.float32),
            x_query.astype(np.float32),
            y_query.astype(np.float32),
        )

    def __len__(self):
        return 1000000
