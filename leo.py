import torch


class LSTMTheta(torch.nn.Module):
    def __init__(self, input_size, hidden_size, horizon):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizon = horizon

        self.lstm = torch.nn.LSTMCell(input_size, hidden_size, bias=True)
        self.linear = torch.nn.Linear(hidden_size, input_size, bias=True)

    def forward(self, inputs):
        tasks_x = inputs

        # num_points is 2 for 2D dataset, and equal to input_size.
        num_tasks, num_sequences, num_frames, num_points = tasks_x.shape

        output = []
        for x in tasks_x:
            h = torch.zeros(num_sequences, self.hidden_size)
            c = torch.zeros(num_sequences, self.hidden_size)

            for i in range(x.shape[1]):
                h, c = self.lstm(x[:, i], (h, c))

            pred = self.linear(h)

            task_output = [pred]
            for i in range(self.horizon - 1):
                h, c = self.lstm(pred, (h, c))
                pred = self.linear(h)
                task_output.append(pred)
            assert len(task_output) == self.horizon
            task_output = torch.stack(task_output, dim=1)
            output.append(task_output)

        out = torch.stack(output, dim=0)
        assert out.shape == (
            num_tasks,
            num_sequences,
            self.horizon,
            num_points,
        ), out.shape
        return out


class LEO:
    """Trains and assesses a sequential LEO model."""

    def __init__(self, encoder, relation_net, decoder, f_theta):
        self.enconder = encoder
        self.relation_net = relation_net
        self.decoder = decoder
        self.t_theta = f_theta

        params = (
            list(self.encoder.parameters())
            + list(self.relation_net.parameters())
            + list(self.decoder.parameters())
            + list(self.f_theta.parameters())
        )
        self.optimizer = torch.optim.Adam(list(self.enconder.para))

    def _forward(self, inputs, parameters):
        mean_z, variance_z = self.relation_net(self.enconder(inputs))
        z = mean_z + torch.randn() * torch.sqrt(variance_z)
        mean_theta, variance_theta = self.decoder(z)
        theta = mean_theta + torch.randn() * torch.sqrt(variance_theta)
        outputs = self.f_theta(inputs, theta)
        return outputs

    def _train_loop(self, task, train):
        x_support, y_support, x_query, y_query = task

        mean_z, variance_z = self.relation_net(self.enconder(x_support))
        z = mean_z + torch.randn() * torch.sqrt(variance_z)
        z_dashed = z.clone()
        mean_theta, variance_theta = self.decoder(z_dashed)
        theta = mean_theta + torch.randn() * torch.sqrt(variance_theta)
        outputs = self.f_theta(x_support, theta)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, y_support)

        grads = torch.autograd.grad(
            loss,
            z_dashed,
            create_graph=train,
        )

        z_dashed = z_dashed - self.inner_lr * grads[0].grad
        assert len(grads) == 1

        mean_theta, variance_theta = self.decoder(z_dashed)
        theta = mean_theta + torch.randn() * torch.sqrt(variance_theta)
        outputs = self.f_theta(x_query, theta)
        loss = criterion(outputs, y_query)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
