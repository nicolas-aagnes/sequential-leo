import torch


class LSTMLinear(torch.nn.Module):
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
