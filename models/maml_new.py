import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass

from torch.nn.parameter import Parameter

from models.base_maml import BaseMAML

torch.autograd.set_detect_anomaly(True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MAMLConfig:
    input_size: int
    hidden_size: int
    num_timesteps_pred: int


def functional_lstm_cell(x, h, c, parameters):
    """Refer to the pytorch LSTM and LSTM docs for these equations."""
    # TODO: These values are hardcoded for the MAML Sine2D case.
    # The hardcoded 1 is due to the num_samples = 1 in the config.

    # Unsqueeze for batch multiplication
    num_samples, input_size = x.shape
    x = x.unsqueeze(-1)
    h = h.unsqueeze(-1)
    c = c.unsqueeze(-1)
    assert x.shape[1:] == (2, 1), x.shape
    assert h.shape[1:] == (16, 1), h.shape
    assert c.shape[1:] == (16, 1), c.shape
    assert parameters["Whi"].shape == (16, 16), parameters["Whi"].shape

    input = torch.sigmoid(
        torch.bmm(parameters["Wii"].expand(num_samples, *parameters["Wii"].shape), x)
        + parameters["bii"].expand(num_samples, *parameters["bii"].shape)
        + torch.bmm(parameters["Whi"].expand(num_samples, *parameters["Whi"].shape), h)
        + parameters["bhi"].expand(num_samples, *parameters["bhi"].shape)
    )
    forget = torch.sigmoid(
        torch.bmm(parameters["Wif"].expand(num_samples, *parameters["Wif"].shape), x)
        + parameters["bif"].expand(num_samples, *parameters["bif"].shape)
        + torch.bmm(parameters["Whf"].expand(num_samples, *parameters["Whf"].shape), h)
        + parameters["bhf"].expand(num_samples, *parameters["bhf"].shape)
    )
    gate = torch.tanh(
        torch.bmm(parameters["Wig"].expand(num_samples, *parameters["Wig"].shape), x)
        + parameters["big"].expand(num_samples, *parameters["big"].shape)
        + torch.bmm(parameters["Whg"].expand(num_samples, *parameters["Whg"].shape), h)
        + parameters["bhg"].expand(num_samples, *parameters["bhg"].shape)
    )
    output = torch.sigmoid(
        torch.bmm(parameters["Wio"].expand(num_samples, *parameters["Wio"].shape), x)
        + parameters["bio"].expand(num_samples, *parameters["bio"].shape)
        + torch.bmm(parameters["Who"].expand(num_samples, *parameters["Who"].shape), h)
        + parameters["bho"].expand(num_samples, *parameters["bho"].shape)
    )

    cell_state = forget * c + input * gate
    hidden_state = output * torch.tanh(cell_state)

    return hidden_state.squeeze(-1), cell_state.squeeze(-1)


def generate_lstm_cell_params():
    parameters = {}

    for type1 in ("i", "h"):
        for type2 in ("i", "f", "g", "o"):
            parameters[f"W{type1}{type2}"] = torch.nn.init.xavier_uniform_(
                torch.empty(
                    16, 2 if type1 == "i" else 16, requires_grad=True, device=DEVICE
                )
            )

    for type1 in ("i", "h"):
        for type2 in ("i", "f", "g", "o"):
            parameters[f"b{type1}{type2}"] = torch.nn.init.zeros_(
                torch.empty(16, 1, requires_grad=True, device=DEVICE)
            )

    parameters["linear_output"] = torch.nn.init.xavier_uniform_(
        torch.empty(2, 16, requires_grad=True, device=DEVICE)
    )

    return parameters


class MAML(BaseMAML):
    """Trains and assesses a sequential LEO model."""

    def __init__(
        self, num_inner_steps, inner_lr, learn_inner_lr, outer_lr, log_dir, config
    ):
        super().__init__(num_inner_steps, inner_lr, learn_inner_lr, outer_lr, log_dir)
        self.config = config

        self.meta_parameters = generate_lstm_cell_params()

        self.optimizer = torch.optim.Adam(
            params=list(self.meta_parameters.values()), lr=outer_lr
        )

    def _inner_loss(self, predictions, target):
        return F.mse_loss(predictions, target)

    def _outer_loss(self, predictions, target, parameters):
        return F.mse_loss(predictions, target)

    def _forward(self, x, parameters):
        # x has shape (num_samples, timesteps, input_size)
        num_samples, timesteps, input_size = x.shape
        assert input_size == self.config.input_size

        h = torch.zeros(num_samples, self.config.hidden_size, device=DEVICE)
        c = torch.zeros(num_samples, self.config.hidden_size, device=DEVICE)

        for i in range(timesteps):
            h, c = functional_lstm_cell(x[:, i], h, c, parameters)

        assert h.shape == (num_samples, self.config.hidden_size)

        pred = torch.bmm(
            parameters["linear_output"].expand(
                num_samples, *parameters["linear_output"].shape
            ),
            h.unsqueeze(-1),
        ).squeeze(-1)
        assert pred.shape == (num_samples, self.config.input_size), pred.shape

        predictions = [pred]
        for _ in range(self.config.num_timesteps_pred - 1):
            h, c = functional_lstm_cell(predictions[-1], h, c, parameters)
            pred = torch.bmm(
                parameters["linear_output"].expand(
                    num_samples, *parameters["linear_output"].shape
                ),
                h.unsqueeze(-1),
            ).squeeze(-1)
            predictions.append(pred)

        assert len(predictions) == self.config.num_timesteps_pred
        predictions = torch.stack(predictions, dim=1)
        assert predictions.shape == (
            num_samples,
            self.config.num_timesteps_pred,
            input_size,
        ), predictions.shape

        return predictions

    def _inner_loop(self, x_support, y_support, train):
        # Get shapes for sanity checking.
        assert len(x_support.shape) == len(y_support.shape) == 3
        assert (
            x_support.shape[0] == y_support.shape[0]
            and x_support.shape[2] == y_support.shape[2]
            and x_support.shape[2] == self.config.input_size
        )

        # Clone parameters.
        parameters = {k: torch.clone(v) for k, v in self.meta_parameters.items()}

        # Optimize inner loop.
        for _ in range(self.num_inner_steps):
            predictions = self._forward(x_support, parameters)
            assert predictions.shape == y_support.shape
            loss = self._inner_loss(predictions, y_support)

            # Calculate gradients wrt parameters.
            grads = torch.autograd.grad(
                loss,
                parameters.values(),
                create_graph=train,
            )

            # Gradient update on parameters.
            for k, grad in zip(parameters.keys(), grads):
                assert grad.shape == parameters[k].shape
                parameters[k] = parameters[k] - self.inner_lr * grad

        return parameters


class MAMLOLD(BaseMAML):
    """Trains and assesses a sequential LEO model."""

    def __init__(
        self, num_inner_steps, inner_lr, learn_inner_lr, outer_lr, log_dir, config
    ):
        super().__init__(num_inner_steps, inner_lr, learn_inner_lr, outer_lr, log_dir)
        self.config = config

        self.lstm = torch.nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_cell = torch.nn.LSTMCell(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(
            in_features=config.hidden_size, out_features=config.input_size
        )

        self.lstm.to(DEVICE)
        self.lstm_cell.to(DEVICE)
        self.linear.to(DEVICE)

        self.meta_parameters = {
            **dict(self.lstm.named_parameters()),
            **dict(self.linear.named_parameters()),
        }

        self.optimizer = torch.optim.Adam(
            params=list(self.meta_parameters.values()), lr=outer_lr
        )

    def _inner_loss(self, predictions, target):
        return F.mse_loss(predictions, target)

    def _outer_loss(self, predictions, target, parameters):
        return F.mse_loss(predictions, target)

    def _forward(self, x, parameters):
        # x has shape (num_samples, timesteps, input_size)
        num_samples, timesteps, input_size = x.shape

        # Set the parameters in the lstm and linear models.
        parameters = {
            k: torch.nn.parameter.Parameter(v) for k, v in self.meta_parameters.items()
        }
        for name, _ in self.lstm.named_parameters():
            setattr(self.lstm, name, parameters[name])
        for name, _ in self.linear.named_parameters():
            setattr(self.linear, name, parameters[name])

        output, (h, c) = self.lstm(x)

        pred = self.linear(output[:, -1])
        predictions = [pred[:, None]]

        for _ in range(self.config.num_timesteps_pred - 1):
            output, (h, c) = self.lstm(predictions[-1], (h, c))
            pred = self.linear(output)
            assert pred.shape == predictions[-1].shape
            predictions.append(pred)

        assert len(predictions) == self.config.num_timesteps_pred
        predictions = torch.stack(predictions, dim=1).squeeze(2)
        assert predictions.shape == (
            num_samples,
            self.config.num_timesteps_pred,
            input_size,
        ), predictions.shape

        return predictions

    def _inner_loop(self, x_support, y_support, train):
        # Get shapes for sanity checking.
        assert len(x_support.shape) == len(y_support.shape) == 3
        assert (
            x_support.shape[0] == y_support.shape[0]
            and x_support.shape[2] == y_support.shape[2]
            and x_support.shape[2] == self.config.input_size
        )

        # Clone parameters.
        parameters = {k: torch.clone(v) for k, v in self.meta_parameters.items()}

        # Optimize inner loop.
        for _ in range(self.num_inner_steps):
            predictions = self._forward(x_support, parameters)
            assert predictions.shape == y_support.shape
            loss = self._inner_loss(predictions, y_support)

            # Calculate gradients wrt parameters.
            grads = torch.autograd.grad(
                loss,
                parameters.values(),
                create_graph=train,
            )

            # Gradient update on parameters.
            for k, grad in zip(parameters.keys(), grads):
                assert grad.shape == parameters[k].shape
                parameters[k] = parameters[k] - self.inner_lr * grad

        return parameters
