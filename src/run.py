import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import random

import time

from MyData import (
    CustomDataset,
    restrict_dataset_by_labels,
)

from gpt_back_model import GPT2_back_Model
from NN_model import NN_estimator

import math
import numpy as np

from rnc_loss import RnCLoss

from Transformer_model_tokenize import TransformerModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from utils import (
    split_list,
    visualize_tsne,
    obtain_latent_vectors,
    eval_end_to_end_model,
)
from pretrain_function import pre_train
from args_parser import parse_arguments
import os

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from NN_model import NN_estimator
import xgboost as xgb
from utils import  compare_model_parameters

def fix_seeds(seed=0):
    random.seed(seed)                  # Pythonの標準ライブラリの乱数
    np.random.seed(seed)               # NumPyの乱数
    torch.manual_seed(seed)            # PyTorchの乱数 (CPU)
    torch.cuda.manual_seed(seed)       # PyTorchの乱数 (GPU)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


fix_seeds(42)

args = parse_arguments()

vocab_size = args.vocab_size
max_seq_length = args.max_seq_length
d_model = args.d_model
nhead = args.nhead
num_layers = args.num_layers
dim_feedforward = args.dim_feedforward
feature_dim = args.feature_dim
AP_num = args.AP_num
epochs = args.epochs
task_epochs = args.task_epochs
lr_for_pretrain = args.lr_for_pretrain
lr_decay_rate = args.lr_decay_rate
is_sort = args.is_sort
test_masking_rate = args.test_masking_rate
test_fault_rate = args.test_fault_rate
reference_points_ratio = args.reference_points_ratio
model_type = args.model_type
pre_train_method = args.pre_train_method
model_setting = args.model_setting
conduct_pre_train = args.conduct_pre_train
aug_setting = args.aug_setting
auto_confirm = args.auto_confirm
output_layer = args.output_layer

gpt2_random_init = bool(args.model_setting != "pre_trained")

dataset_name = args.dataset_name
is_projection_head = args.is_projection_head

model_name = "{}_{}_{}_{}_{}".format(
    dataset_name, pre_train_method, model_setting, model_type, reference_points_ratio
)

pre_trained_model_path = "./model/{}_100.pth".format(model_name)

print(args)
print(model_name)

if os.path.exists(pre_trained_model_path) and conduct_pre_train:
    print("*" * 20, "Maybe You have ALREADY trained this model", "*" * 20)
    choice = input("Do you want to continue? (yes/no): ").lower()
    if choice == "yes" or auto_confirm:
        print("Great! Let's continue.")
    else:
        print("Goodbye!")
        exit()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

if model_type == "transformer":
    model = TransformerModel(
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        feature_dim,
    ).to(device)
elif model_type == "gpt2":
    model = GPT2_back_Model(
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        feature_dim,
        will_recover_rssi=bool("bert" in pre_train_method),
        random_init=gpt2_random_init,
        is_projection_head=is_projection_head,
    ).to(device)
elif model_type == "NN":
    model = NN_estimator(
        vocab_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        feature_dim,
        seq_len=AP_num,
    ).to(device)


# Train the model
# train_transformer_model(dataloader, model, epochs=10, lr=1e-3)

if dataset_name == "hamada":
    csv_file = (
        "/src/hamada_dataset/Osaka_u/train/data_with_id.csv"  # Path to your CSV file
    )
elif dataset_name == "uji":
    csv_file = "/src/uji_indoor/TrainingData.csv"  # Path to your CSV file
elif dataset_name == "hamada_cell":
    csv_file = "/src/celluer_dataset/train_dataset.csv"  # Path to your CSV file


train_dataset = CustomDataset(
    csv_file,
    is_sorted=is_sort,
    data_name=dataset_name,
    AP_num=AP_num,
    dataset_type="train",
    fault_ratio=0.0,
)


if dataset_name == "hamada":
    test_csv_file = (
        "/src/hamada_dataset/Osaka_u/test/data_with_id.csv"  # Path to your CSV file
    )
elif dataset_name == "uji":
    test_csv_file = "/src/uji_indoor/TrainingData.csv"  # Path to your CSV file
elif dataset_name == "hamada_cell":
    test_csv_file = "/src/celluer_dataset/test_dataset.csv"  # Path to your CSV file


test_dataset = CustomDataset(
    test_csv_file,
    is_sorted=is_sort,
    data_name=dataset_name,
    AP_num=AP_num,
    dataset_type="test",
    fault_ratio=test_fault_rate,
    mask_rate=test_masking_rate,
)

labels = train_dataset.get_unique_label()

# 0.7
reference_points, test_points = split_list(labels, 0.7)
# reference_points, test_points = split_list(labels, 1.0)
print(
    "All point: {} training point: {} Test point: {}".format(
        len(labels), len(reference_points), len(test_points)
    )
)
reference_points, _ = split_list(reference_points, reference_points_ratio)

if dataset_name == "hamada_cell":
    test_points = test_dataset.get_unique_label()

print(reference_points)
print(test_points)

print(
    "All point: {} Actual training point: {}  Test point: {}".format(
        len(labels), len(reference_points), len(test_points)
    )
)

train_dataset = restrict_dataset_by_labels(
    train_dataset, reference_points, num_per_label=100
)
test_dataset = restrict_dataset_by_labels(
    test_dataset, test_points, num_per_label=10000
)

# train_dataset, test_dataset = split_dataset_by_labels(dataset, reference_points)
# train_dataset, test_dataset = split_dataset_by_labels_for_few_shot(
#     dataset, reference_points, num_per_label=5
# )

# Train and test DataLoaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)



criterion = RnCLoss(temperature=1)
optimizer_for_pretrain = optim.Adam(model.parameters(), lr=lr_for_pretrain)


encoded_features_array = []
labels_array = []

is_mask = True
if aug_setting == "noise":
    is_mask = False


if conduct_pre_train:
    model = pre_train(
        model,
        epochs,
        train_loader,
        optimizer_for_pretrain,
        criterion,
        model_name,
        device,
        pretrain_method=pre_train_method,
        is_mask=is_mask,
    )

    if pre_train_method == "rnc":
            model = pre_train(
            model,
            epochs,
            train_loader,
            optimizer_for_pretrain,
            criterion,
            model_name,
            device,
            pretrain_method="end-to-end",
            is_mask=is_mask,
        )

import copy

model.load_state_dict(torch.load(pre_trained_model_path, weights_only=True))

# Fine-tuning
lr = 0.01


optimizer_for_task = optim.Adam(model.parameters(), lr=lr)
# -----------------------------------
eval_end_to_end_model(model, test_loader, device)

if pre_train_method == "end-to-end":
    is_test = False
    latent_vectors_array, labels_array = obtain_latent_vectors(
        model, train_loader, args.is_shallow, is_test, device
    )
    start = time.perf_counter()
    eval_end_to_end_model(model, test_loader, device)
    end = time.perf_counter()
    print("Time:", '{:.4f}'.format((end-start)/len(test_loader)*1000))

else:
    is_test = False
    
    latent_vectors_array, labels_array = obtain_latent_vectors(
        model, train_loader, args.is_shallow, is_test, device
    )

    latent_vectors = np.vstack(latent_vectors_array)
    # latent_vectors = np.vstack([latent_vectors, latent_vectors_array[-1]])

    print("Latent vector size: ", latent_vectors.shape)

    labels = np.vstack(labels_array)
    print("Label size: ", labels.shape)

    # ランダムフォレスト回帰モデルの訓練
    if output_layer == "RF":
        out_model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2) 
    else:
        out_model = MLPRegressor(hidden_layer_sizes=(256,), max_iter=2000, random_state=42)

    out_model.fit(latent_vectors, labels)
    is_test = True
    test_latent_vectors, test_labels_array = obtain_latent_vectors(
        model, test_loader, args.is_shallow, is_test, device
    )

    test_latent_vectors = np.vstack(test_latent_vectors)
    test_labels = np.vstack(test_labels_array)

    print(test_labels.shape)
    print(test_latent_vectors.shape)


    start = time.perf_counter()
    predictions = out_model.predict(test_latent_vectors)
    end = time.perf_counter()
    print("Time:", '{:.2f}'.format((end-start)/len(train_loader)*1000))

    mse = math.sqrt(
        mean_squared_error(test_labels, predictions, multioutput="uniform_average")
    )
    print(f"Mean Squared Error: {mse:.4f}")

# else:

#     for epoch in range(task_epochs):
#         model.train()
#         total_loss = 0

#         for (
#             id_input,
#             rssi_input,
#             labels,
#             noised_rssi_input_0,
#             noised_rssi_input_1,
#             at_mask_0,
#             at_mask_1,
#         ) in train_loader:
#             # Move data to the same device as the model
#             # print(rssi_input)
#             # print(labels)

#             B, L = id_input.size(0), id_input.size(1)
#             id_input = id_input.to(device)

#             rssi_input = rssi_input.view(B, 1, L).to(device)

#             labels = labels.to(device)

#             optimizer_for_task.zero_grad()

#             # Forward pass
#             outputs = model(id_input, rssi_input)
#             # outputs = model(id_input, rssi_input)
#             loss = criterion_for_task(outputs, labels)

#             # Backward pass and optimization
#             loss.backward()
#             optimizer_for_task.step()

#             total_loss += math.sqrt(loss.item())

#         print(
#             f"Epoch [{epoch + 1}/{task_epochs}], Loss: {total_loss / len(train_loader):.4f}"
#         )


#     encoded_features_array = []
#     labels_array = []

#     model.eval()

#     for (
#         id_input,
#         rssi_input,
#         labels,
#         noised_rssi_input_0,
#         noised_rssi_input_1,
#         at_mask_0,
#         at_mask_1,
#     ) in test_loader:

#         # Move data to the same device as the model
#         B, L = id_input.size(0), id_input.size(1)
#         id_input = id_input.to(device)
#         rssi_input = rssi_input.view(B, 1, L).to(device)
#         labels = labels.to(device)

#         # Forward pass
#         # outputs, encoded_features = model(id_input, rssi_input)
#         outputs = model(id_input, rssi_input)

#         # Compute loss
#         loss = criterion_for_task(outputs, labels)
#         total_loss += math.sqrt(loss.item())

#     print(f"Eval[cm]: {total_loss / len(test_loader):.4f}")
