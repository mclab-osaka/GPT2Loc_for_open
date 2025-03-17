import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import random
import torch
import torch.nn as nn
import math
import matplotlib.patches as mpatches


import matplotlib.colors as mcolors
import numpy as np

import time

def generate_cmap(colors, cmap_name = 'custom_cmap'):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for vi, ci in zip(values, colors):
        color_list.append( ( vi/ vmax, ci) )

    return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, 256)




def split_list(lst, ratio):
    # リストをランダムにシャッフル
    random.shuffle(lst)

    # 分割位置を計算
    split_idx = int(len(lst) * ratio)

    # リストを指定した割合で分割
    list1 = lst[:split_idx]
    list2 = lst[split_idx:]

    return list1, list2


def assign_ranks(arr):
    # ユニークな数字をソートする
    unique_vals = np.unique(arr)
    # 各ユニークな数字に順番を割り振る（インデックスとして）
    ranks = {val: idx for idx, val in enumerate(unique_vals)}
    # 元の配列の各値を対応するランクに変換する
    return np.array([ranks[val] for val in arr])

def visualize_tsne(features, labels, pos=0, method="t-sne", prefix="gpt2"):
    features = features.cpu().detach().numpy()  # [B, F]
    labels = labels.cpu().detach().numpy()

    plt.rcParams["font.size"] = 16

    if method == "t-sne":
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(features)  # [B, 2] に次元削減

        # 色付けに使用するラベルの一次元目 (例: labels[:, 0])
        color_labels = labels[:, pos]

        discrete_meter = 300

        label_categories = color_labels // discrete_meter # 500 ごとにカテゴリ化
        label_categories = label_categories.astype(int)
        # label_categories = assign_ranks(label_categories)

        label_unique = list(np.unique(label_categories))

        num_categories = 10
        num_categories = max(label_categories) + 1
        

        cmap = plt.get_cmap("tab10", num_categories)
        # cmap = generate_cmap(['#1c3f75', '#068fb9','#f1e235', '#d64e8b', '#730e22'], 'cmthermal')
        colors = cmap(label_categories)
        print(colors)

        # cmap = cm.Blues
        # norm = Normalize(vmin=min(color_labels), vmax=max(color_labels))
        # colors = cmap(norm(color_labels))

        # t-SNEのプロット
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            color=colors,
            s=50,
            alpha=0.7,
        )

        if pos == 0:
            loc = "X"
        else:
            loc = "Y"

        # plt.colorbar(scatter, label="Label Dimension 1")  # カラーバー
        legend_patches = [
            mpatches.Patch(
                color=cmap(i), label=f"{i*discrete_meter}-{(i+1)*discrete_meter-1}"
            )
            for i in range(num_categories) if i in label_unique
        ]
        # plt.legend(handles=legend_patches, title="Label Range")
        # plt.title("t-SNE Visualization of Features")
        # plt.xlabel("t-SNE Dimension 1")
        # plt.ylabel("t-SNE Dimension 2")
        plt.savefig("./result_images/latent_space_{}_{}.pdf".format(prefix, pos))
        plt.close()
    else:
        tsne = TSNE(n_components=1, random_state=42)
        reduced_features = tsne.fit_transform(features)  # [B, 2] に次元削減

        xy_coords = labels
        tsne_1d = reduced_features.flatten()
        # グリッドを生成
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, 3000, 100),  # x方向の格子点
            np.linspace(0, 1500, 100),  # y方向の格子点
        )

        # print(tsne_1d)
        # print(grid_x)

        # 格子上に値を補間
        grid_z = griddata(xy_coords, tsne_1d, (grid_x, grid_y), method="cubic")

        # コンター図のプロット
        plt.clf()
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap="viridis")
        plt.colorbar(contour, label="t-SNE Reduced Value (1D)")
        plt.title("t-SNE Reduced Features as Contour Map")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.savefig("radio_map_{}.pdf".format(pos))
        plt.close()


def obtain_latent_vectors(model, train_loader, is_shallow, is_test, device):
    # 潜在変数とラベルを収集
    latent_vectors = []
    labels_array = []

    t_latent_vectors = []
    t_labels_array = []

    model.eval()
    for (
        id_input,
        rssi_input,
        labels,
        noised_rssi_input_0,
        noised_rssi_input_1,
        at_mask_0,
        at_mask_1,
        label_0,
        label_1,
    ) in train_loader:

        B, L = id_input.size(0), id_input.size(1)
        id_input = id_input.to(device)
        _rssi_input = rssi_input.view(B, 1, L).to(device)
        at_mask_0 = at_mask_0.to(device)

        labels = labels.to(device)

        if not is_test:
            at_mask_0 = None

        # Forward pass
        if is_shallow:
            latent = rssi_input
        else:
            start = time.perf_counter()
            _, latent, _ = model(id_input, _rssi_input, attention_mask=at_mask_0)

            end = time.perf_counter()
            print("Infer Time:", '{:.4f}'.format((end-start)/256*1000))

        latent_vectors.append(latent.detach().cpu().numpy())
        labels_array.append(labels.detach().cpu().numpy())

        t_latent_vectors.append(latent.detach().cpu())
        t_labels_array.append(labels.detach().cpu())

    _encoded_features = torch.cat(t_latent_vectors, dim=0)
    _labels = torch.cat(t_labels_array, dim=0)

    print(_encoded_features.size())

    prefix = "training"
    if is_test:
        prefix = "test"
    visualize_tsne(_encoded_features, _labels, pos=0, prefix=prefix)
    visualize_tsne(_encoded_features, _labels, pos=1, prefix=prefix)

    return latent_vectors, labels_array


def eval_end_to_end_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    criterion_for_task = nn.MSELoss()

    for (
        id_input,
        rssi_input,
        labels,
        noised_rssi_input_0,
        noised_rssi_input_1,
        at_mask_0,
        at_mask_1,
        label_0,
        label_1,
    ) in test_loader:

        # Move data to the same device as the model
        B, L = id_input.size(0), id_input.size(1)
        id_input = id_input.to(device)
        rssi_input = rssi_input.view(B, 1, L).to(device)
        labels = labels.to(device)
        at_mask_0 = at_mask_0.to(device)

        # Forward pass
        # outputs, encoded_features = model(id_input, rssi_input)
        outputs, *_ = model(id_input, rssi_input, attention_mask=at_mask_0)
        outputs = outputs.squeeze()

        # Compute loss
        loss = criterion_for_task(outputs, labels)
        total_loss += math.sqrt(loss.item())

    print(f"Eval[m]: {total_loss / len(test_loader):.4f}")


import torch


def compare_model_parameters(model1, model2, verbose=True):
    """
    2つのPyTorchモデルのパラメータがすべて同じであるかを確認する関数。

    Args:
        model1 (torch.nn.Module): 比較対象の1つ目のモデル
        model2 (torch.nn.Module): 比較対象の2つ目のモデル
        verbose (bool): 異なるパラメータがあった場合に詳細を表示するか

    Returns:
        bool: すべてのパラメータが一致すれば True、一致しなければ False
    """
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if name1 != name2:
            if verbose:
                print(f"異なるパラメータ名: {name1} != {name2}")
            return False

        if not torch.equal(param1, param2):
            if verbose:
                diff = torch.abs(param1 - param2).max().item()
                print(f"パラメータ '{name1}' が異なります (最大差: {diff})")
            return False

    return True
