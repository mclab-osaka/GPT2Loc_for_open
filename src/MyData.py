import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import random
from torch.utils.data import Subset


class CustomDataset(Dataset):
    def __init__(
        self,
        csv_file,
        AP_num=137,
        is_sorted=True,
        data_name="hamada",
        dataset_type="train",
        fault_ratio=0.0,
        mask_rate=0.1,
    ):
        # Load CSV data
        self.data = pd.read_csv(csv_file)
        self.mask_rate = mask_rate

        # Extract MAC address columns, time, x, y
        if data_name == "hamada":
            self.rssi_data = self.data.iloc[:, :AP_num].values  # RSSI values
            self.x_data = self.data["x"].values  # X coordinate
            self.y_data = self.data["y"].values  # Y coordinate


        elif data_name == "hamada_cell":
            self.rssi_data = self.data.iloc[:, :AP_num].values  # RSSI values
            # self.rssi_data[self.rssi_data < 0.25] = 0
            self.x_data = self.data["x"].values  # X coordinate
            self.x_data = self.x_data - self.x_data.min()
            self.y_data = self.data["y"].values  # Y coordinate
            self.y_data = self.y_data - self.y_data.min()


        elif data_name == "uji":
            target_floor = 0
            target_building = 2
            if dataset_type == "train":
                self.data = self.data[
                    (self.data["FLOOR"] == target_floor)
                    & (self.data["BUILDINGID"] == target_building)
                ]
            else:
                self.data = self.data[
                    (self.data["FLOOR"] == target_floor)
                    & (self.data["BUILDINGID"] == target_building)
                ]

            self.rssi_data = self.data.iloc[:, :AP_num].values  # RSSI values
            self.x_data = self.data["LONGITUDE"].values  # X coordinate
            self.y_data = self.data["LATITUDE"].values  # Y coordinate

            self.rssi_data[self.rssi_data == 100] = 0
            self.rssi_data = self.rssi_data * -1
            self.rssi_data = (self.rssi_data - self.rssi_data.min()) / (
                self.rssi_data.max() - self.rssi_data.min()
            )
            self.x_data = self.x_data - self.x_data.min()
            self.y_data = self.y_data - self.y_data.min()

            self.x_data = self.x_data.astype(int)
            self.y_data = self.y_data.astype(int)

        self.unique_label = list(set(zip(self.x_data, self.y_data)))

        # Create a unique ID for each MAC address
        self.mac_ids = np.arange(AP_num)

        self.is_sorted = is_sorted
        self.AP_num = AP_num
        self.dataset_type = dataset_type
        self.fault_ratio = fault_ratio

    def __len__(self):
        return len(self.data)

    def get_unique_label(self):
        return self.unique_label

    def __getitem__(self, idx):
        # Get RSSI values for the current sample
        rssi_values = self.rssi_data[idx]
        rssi_values_0 = self.rssi_data[idx]
        rssi_values_1 = self.rssi_data[idx]

        fault_indicies_0 = random.sample(
            range(self.AP_num), int(self.AP_num * self.fault_ratio)
        )
        fault_indicies_1 = random.sample(
            range(self.AP_num), int(self.AP_num * self.fault_ratio)
        )

        rssi_values_0[fault_indicies_0] = 0
        rssi_values_1[fault_indicies_1] = 0

        # Sort by RSSI values in descending order
        # sorted_indices = np.array(range(0, self.AP_num))

        sorted_indices = np.array(range(0, self.AP_num))
        if self.is_sorted:
            sorted_indices = np.argsort(-rssi_values)  # Sort in descending order
            sorted_indices_0 = np.argsort(-rssi_values_0)  # Sort in descending order
            sorted_indices_1 = np.argsort(-rssi_values_1)  # Sort in descending order

        sorted_rssi = rssi_values[sorted_indices]
        sorted_ids = self.mac_ids[sorted_indices]

        # Convert to tensors
        sorted_ids_tensor = torch.tensor(sorted_ids, dtype=torch.long)
        sorted_rssi_tensor = torch.tensor(sorted_rssi, dtype=torch.float32)

        noised_rssi_tensor_0 = (
            sorted_rssi_tensor + torch.randn_like(sorted_rssi_tensor) * 0.05
        )
        noised_rssi_tensor_1 = (
            sorted_rssi_tensor + torch.randn_like(sorted_rssi_tensor) * 0.05
        )

        # print(noised_rssi_tensor_0)
        # print(noised_rssi_tensor_1)

        input_length = sorted_rssi_tensor.size(0)
        mask_percentage = self.mask_rate

        num_to_mask = int(input_length * mask_percentage)

        mask_indices_0 = np.random.choice(input_length, num_to_mask, replace=False)
        mask_indices_1 = np.random.choice(input_length, num_to_mask, replace=False)

        attention_mask_0 = np.ones(input_length, dtype=int)
        attention_mask_1 = np.ones(input_length, dtype=int)

        attention_mask_0[mask_indices_0] = 0
        attention_mask_1[mask_indices_1] = 0

        if self.is_sorted:
            sorted_ids_tensor = sorted_ids_tensor
            sorted_rssi_tensor = sorted_rssi_tensor

        # Get label (x, y coordinates)
        label = torch.tensor([self.x_data[idx], self.y_data[idx]], dtype=torch.float32)
        label_0 = label + torch.randn_like(label) * 0.5
        label_1 = label + torch.randn_like(label) * 0.5

        return (
            sorted_ids_tensor,
            sorted_rssi_tensor,
            label,
            noised_rssi_tensor_0,
            noised_rssi_tensor_1,
            attention_mask_0,
            attention_mask_1,
            label_0,
            label_1,
        )


def restrict_dataset_by_labels(dataset, train_positions, num_per_label):
    train_indices = []
    train_label_counter = {p: 0 for p in train_positions}

    indices = np.random.permutation(len(dataset))
    dataset = Subset(dataset, indices)

    # Identify train and test indices based on labels (x, y positions)
    for idx in range(len(dataset)):
        _, _, label, _, _, _, _, _, _ = dataset[idx]
        position = tuple(label.numpy())
        if (
            position in train_positions
            and train_label_counter[position] < num_per_label
        ):
            train_indices.append(idx)
            train_label_counter[position] += 1

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    return train_dataset


if __name__ == "__main__":

    csv_file = (
        "/src/hamada_dataset/Osaka_u/train/data_with_id.csv"  # Path to your CSV file
    )
    # csv_file = "/src/ujiindoorloc/UJIndoorLoc/train/trainingData.csv"
    dataset = CustomDataset(csv_file, data_name="hamada")

    # train_dataset, test_dataset = split_dataset_random(dataset, train_ratio=0.8)

    # Label-based split
    labels = dataset.get_unique_label()
    reference_points = random.sample(labels, int(len(labels) * 0.7))
