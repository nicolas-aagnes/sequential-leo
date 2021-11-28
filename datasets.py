import torch
import numpy as np
import time
import numpy as np
from glob import glob
import json
import matplotlib.pyplot as plt
import argparse
import torch
import time


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
        amplitude = np.random.uniform(0.5, 5)
        phase = np.random.uniform(0.5, np.pi)

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


def get_data_file(annotations_path, subject_id):
    return glob(annotations_path + "/*subject" + str(subject_id) + "_data*")[0]


def get_coords_file(annotations_path, subject_id):
    return glob(annotations_path + "/*subject" + str(subject_id) + "_joint_3d*")[0]


def get_subject_data_length(annotations_path, subject_id):
    data_file = get_data_file(annotations_path, subject_id)

    ## Parse data file
    with open(data_file, "r") as data_json_file:
        data = json.load(data_json_file)

    return len(data["images"])


def get_subject_action_boundaries(annotations_path, subject_id):
    data_file = get_data_file(annotations_path, subject_id)

    ## Parse data file
    with open(data_file, "r") as data_json_file:
        data = json.load(data_json_file)

    boundaries = [0]
    prev_action = data["images"][0]["action_name"]
    for i, item in enumerate(data["images"]):
        if i == 0:
            continue
        if item["action_name"] != prev_action:
            prev_action = item["action_name"]
            boundaries.append(i)

    boundaries.append(len(data["images"]))
    # print(boundaries)

    return boundaries


class Human36M(torch.utils.data.Dataset):
    def __init__(
        self,
        num_support,
        num_query,
        num_timesteps=50,
        num_timesteps_pred=10,
        data_path="./annotations",
    ):
        self.num_support = num_support
        self.num_query = num_query
        self.num_timesteps = num_timesteps
        self.num_timesteps_pred = num_timesteps_pred
        self.data_path = data_path

        start_time = time.time()

        # Get subject data lengths.
        self.subject_ids = [1, 6, 7, 8, 9, 11]  # 5 is ignored as it is for testing.
        self.subject_data_lengths = {
            id: get_subject_data_length(data_path, id) for id in self.subject_ids
        }
        self.subject_action_boundaries = {
            id: get_subject_action_boundaries(data_path, id) for id in self.subject_ids
        }

        subject_data_file = {
            id: get_data_file(data_path, id) for id in self.subject_ids
        }
        self.subject_data = {}
        for id, subject_data_file in subject_data_file.items():
            with open(subject_data_file, "r") as subject_data_file:
                self.subject_data[id] = json.load(subject_data_file)

        self.subject_data, self.subject_coords = {}, {}
        for subject_id in self.subject_ids:
            data_file = get_data_file(data_path, subject_id)
            coords_file = get_coords_file(data_path, subject_id)

            ## Parse data file
            with open(data_file, "r") as data_json_file:
                data = json.load(data_json_file)

            ## Parse coordinates file
            with open(coords_file, "r") as coords_json_file:
                coords = json.load(coords_json_file)

            self.subject_data[subject_id] = data
            self.subject_coords[subject_id] = coords

        print("Data loading took", time.time() - start_time)

    def __getitem__(self, index):
        """Sample a task for the H36M dataset.
        Returns:
            - x_support: Tensor of shape (num_support, num_timesteps, 2)
            - y_support: Tensor of shape (num_support, num_timesteps_pred,    2)
            - x_query:   Tensor of shape (num_query,   num_timesteps, 2)
            - y_query:   Tensor of shape (num_query,   num_timesteps_pred,    2)
            - subject_id, index: Tuple for plotting the ground truth human pose
        """

        train_data, train_label, query_data, query_label = self.generate_random_task(
            self.num_timesteps,
            self.num_timesteps_pred,
            self.num_support,
            self.num_query,
            self.data_path,
        )

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

    def generate_random_task(
        self, timesteps, timesteps_pred, num_support, num_query, annotations_path
    ):
        ## Get Random Subject
        subject_id = np.random.choice(self.subject_ids, 1)[0]

        boundaries = self.subject_action_boundaries[subject_id]
        timesteps_total = timesteps + timesteps_pred

        ## Get Random Action
        random_action = np.random.randint(len(boundaries[:-1]))

        ## Get Random Index
        index = np.random.randint(
            boundaries[random_action], boundaries[random_action + 1] - timesteps_total
        )

        ## Get Train and Query Data
        data = []
        label = []
        for item in np.arange(num_support + num_query):
            ## Get Data
            _, _, pose3d = self.get_data(
                subject_id, index, annotations_path, timesteps=timesteps_total
            )
            data.append(pose3d[:timesteps].reshape(timesteps, 51))  # 51 = 17 x 3
            label.append(pose3d[timesteps:].reshape(timesteps_pred, 51))

            ## Calculate next index
            index = np.random.randint(
                boundaries[random_action],
                boundaries[random_action + 1] - timesteps_total,
            )

        ## Return Batch
        train_data = data[:num_support]
        train_label = label[:num_support]
        query_data = data[num_support:]
        query_label = label[num_support:]
        return train_data, train_label, query_data, query_label

    def get_data(self, subject_id, random_index, annotations_path, timesteps=1):
        data = self.subject_data[subject_id]
        coords = self.subject_coords[subject_id]

        frames = np.arange(random_index, random_index + timesteps)
        pose3d = []
        for i, item in enumerate(frames):
            meta_data = data["images"][item]
            coord_data = np.array(
                coords[str(meta_data["action_idx"])][str(meta_data["subaction_idx"])][
                    str(meta_data["frame_idx"])
                ]
            )
            pose3d.append(coord_data)
        pose3d = np.array(pose3d)

        return data, meta_data, pose3d
