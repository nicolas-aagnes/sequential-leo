import torch
import numpy as np
import torch.nn.functional as F

from models.base_maml import BaseMAML
from models.mlp import MLP
from dataclasses import dataclass

torch.autograd.set_detect_anomaly(True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LEOConfig:
    num_support: int
    num_timesteps_pred: int
    input_size: int
    encoder_hidden_size: int
    relation_net_hidden_size: int
    z_dim: int
    decoder_hidden_size: int
    f_theta_hidden_size: int


class LSTMTheta(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_timesteps_pred):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_timesteps_pred = num_timesteps_pred

        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size, bias=True)

    def forward(self, x_and_theta):
        x, theta = x_and_theta

        # x has shape (num_samples, timesteps, input_size)
        num_samples, timesteps, input_size = x.shape
        assert input_size == self.input_size

        h = torch.zeros(num_samples, self.hidden_size, device=DEVICE)
        c = torch.zeros(num_samples, self.hidden_size, device=DEVICE)

        for i in range(timesteps):
            h, c = self.lstm_cell(x[:, i], (h, c))

        # Multiply theta by h to get first prediction.
        # h has shape (num_samples, hidden_size)
        # theta has shape (hidden_size, input_size)
        pred = torch.mm(h, theta)
        assert pred.size() == (num_samples, input_size)

        predictions = [pred]
        for i in range(self.num_timesteps_pred - 1):
            h, c = self.lstm_cell(pred, (h, c))
            pred = torch.mm(h, theta)  # TODO: Check if ReLU should be applied here.
            predictions.append(pred)

        assert len(predictions) == self.num_timesteps_pred
        predictions = torch.stack(predictions, dim=1)
        assert predictions.shape == (num_samples, self.num_timesteps_pred, input_size)

        return predictions


class LEO(BaseMAML):
    """Trains and assesses a sequential LEO model."""

    def __init__(
        self, num_inner_steps, inner_lr, learn_inner_lr, outer_lr, log_dir, config
    ):
        super().__init__(num_inner_steps, inner_lr, learn_inner_lr, outer_lr, log_dir)
        self.config = config

        self.encoder = torch.nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.encoder_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.relation_net = MLP(
            config.num_support * config.encoder_hidden_size,
            config.relation_net_hidden_size,
            config.z_dim * 2,  # mean and variance have dim z_dim.
        )
        self.decoder = MLP(
            config.z_dim,
            config.decoder_hidden_size,
            config.f_theta_hidden_size
            * config.input_size
            * 2,  # need to compute both mean and variance.
        )
        self.f_theta = LSTMTheta(
            config.input_size, config.f_theta_hidden_size, config.num_timesteps_pred
        )

        self.encoder.to(DEVICE)
        self.relation_net.to(DEVICE)
        self.decoder.to(DEVICE)
        self.f_theta.to(DEVICE)

        params = (
            list(self.encoder.parameters())
            + list(self.relation_net.parameters())
            + list(self.decoder.parameters())
            + list(self.f_theta.parameters())
        )
        self.optimizer = torch.optim.Adam(params=params, lr=outer_lr)

    def _inner_loss(self, predictions, target):
        return F.mse_loss(predictions, target)

    def _outer_loss(self, predictions, target, parameters):
        return F.mse_loss(predictions, target)

    # def _outer_loss(self, predictions, target, parameters):
    #     return F.mse_loss(predictions, target) + F.mse_loss(
    #         parameters["z"], parameters["z_dashed"].detach()
    #     )

    def _sample_from_normal(self, mean, log_variance):
        return mean + torch.sqrt(torch.exp(log_variance)) * torch.randn(
            *mean.shape, device=DEVICE
        )

    def _forward(self, x_query, parameters):
        return self.f_theta((x_query, parameters["theta"]))

    def _inner_loop(self, x_support, y_support, train):
        # Get shapes for sanity checking.
        assert len(x_support.shape) == len(y_support.shape) == 3
        assert (
            x_support.shape[0] == y_support.shape[0]
            and x_support.shape[2] == y_support.shape[2]
            and y_support.shape[1] == self.config.num_timesteps_pred
            and x_support.shape[2] == self.config.input_size
        )
        num_support, num_timesteps, input_size = x_support.shape
        num_support, num_timesteps_pred, input_size = y_support.shape

        # Embed all samples and combine them into single vector.
        embeddings, _ = self.encoder(x_support)
        embeddings = embeddings[:, -1].ravel()
        assert embeddings.shape == (num_support * self.config.encoder_hidden_size,)

        # Push embeddings through relation network to get task dependent z params.
        z_params = self.relation_net(embeddings).view(2, -1)
        assert z_params.shape == (2, self.config.z_dim)

        # Sample z.
        z = self._sample_from_normal(z_params[0], z_params[1])
        assert z.shape == (self.config.z_dim,)

        # z needs to be cloned as the original z needs to be kept for backpropagating
        # on the final outer loss.
        z_dashed = z.clone()
        if train:
            z_dashed.retain_grad()

        # Method for sampling theta given z.
        def sample_theta(z_dashed):
            theta_params = self.decoder(z_dashed).view(2, -1)
            assert theta_params.shape == (
                2,
                self.config.f_theta_hidden_size * self.config.input_size,
            )
            theta = self._sample_from_normal(theta_params[0], theta_params[1])
            theta = theta.view(self.config.f_theta_hidden_size, self.config.input_size)
            assert theta.shape == (
                self.config.f_theta_hidden_size,
                self.config.input_size,
            )
            return theta

        # Inner gradient steps.
        for _ in range(self.num_inner_steps):
            # Sample theta from its distribution.
            theta = sample_theta(z_dashed)

            if not train:
                return {"z": z, "z_dashed": z_dashed, "theta": theta}

            # Compute inner loss.
            predictions = self.f_theta((x_support, theta))
            assert predictions.shape == y_support.shape
            loss = self._inner_loss(predictions, y_support)

            # Backprop on z.
            grads = torch.autograd.grad(
                loss,
                z_dashed,
                create_graph=train,
            )

            assert len(grads) == 1
            assert grads[0].shape == z_dashed.shape

            z_dashed = z_dashed - self.inner_lr * grads[0]

        parameters = {"z": z, "z_dashed": z_dashed, "theta": sample_theta(z_dashed)}

        return parameters
