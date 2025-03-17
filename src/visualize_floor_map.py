import pandas as pd
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# CSVファイルを読み込む
df = pd.read_csv(
    "/src/hamada_dataset/Osaka_u/test/data_with_id.csv"
)  # 適宜ファイル名を変更

# X, Y カラムを取得し、ユニークな座標を抽出
unique_points = df[["x", "y"]].drop_duplicates()

unique_points = unique_points.values

print(unique_points)
color_labels = unique_points[:, 1]
discrete_meter = 300

label_categories = color_labels // discrete_meter  # 500 ごとにカテゴリ化
label_categories = label_categories.astype(int)

print(label_categories)

num_categories = max(label_categories) + 1

cmap = plt.get_cmap("tab10", num_categories)
colors = cmap(label_categories)
print(colors)

# 散布図を描画
plt.figure(figsize=(8, 6))
plt.scatter(
    unique_points[:, 0], unique_points[:, 1], color=colors, s=100
)  # s は点の大きさ
plt.xlabel("X")
plt.ylabel("Y")

legend_patches = [
            mpatches.Patch(
                color=cmap(i), label=f"{i*discrete_meter/100}-{(i+1)*discrete_meter/100}"
            )
            for i in range(num_categories) if i in label_categories
        ]
plt.legend(handles=legend_patches, title="Relative Position of Y-axis (m)", loc="upper left")


y_min = (
    int(unique_points[:, 1].min()) // discrete_meter * discrete_meter
)  # 最小値を500の倍数に調整
y_max = (
    int(unique_points[:, 1].max()) // discrete_meter * discrete_meter + discrete_meter
)  # 最大値を500の倍数に調整

for y in range(y_min, y_max + discrete_meter, discrete_meter):
    plt.axhline(y=y, color="gray", linestyle="dashed", linewidth=2)  # 点線を描画

plt.title("Unique (X, Y) Points")

plt.savefig("floor_map_with_val.pdf")

# ===== 1. 背景画像（PNG）を読み込む =====
image_path = "/src/hamada_dataset/Osaka_u/R_floor_plan-withAP.png"  # 画像ファイルのパス
img = cv2.imread(image_path, 0)  # OpenCVで画像読み込み (BGR形式)
bg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCVはBGRなのでRGBに変換

# bg_img = np.rot90(img, 2)
# bg_img[:, :, 3] = 128

# ===== 3. Matplotlib でプロット作成 =====
fig, ax = plt.subplots(figsize=(bg_img.shape[1] / 100, bg_img.shape[0] / 100), dpi=100)

ax.imshow(bg_img)  # 背景画像をプロット
# ax.scatter(unique_points["X"], unique_points["Y"], color="red", s=10, label="Data Points")  # 散布図
ax.scatter(
    unique_points[:, 0], unique_points[:, 1], color=colors, s=3000
)

ax.invert_xaxis()  # X 軸を反転
ax.invert_yaxis()  # Y 軸を反転

plt.legend(handles=legend_patches, title="Relative Position of Y-axis (m)",
            loc="upper left", fontsize=50, title_fontsize=50,
            ncol=2, bbox_transform=fig.transFigure)

# Y軸方向に500ごとの線を引く
for y in range(y_min, y_max + discrete_meter, discrete_meter):
    ax.axhline(y=y, color="gray", linestyle="dashed", linewidth=2)  # 点線を描画

# 軸を非表示にする
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# ===== 4. プロット画像を OpenCV 形式に変換して保存 =====
fig.canvas.draw()
plot_img = np.array(fig.canvas.renderer.buffer_rgba())  # Matplotlib の画像データを取得
plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)  # RGBA → RGB に変換

# resized_image = cv2.resize(plot_img, (800, 600))
cv2.imwrite("output.png", cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))  # 結果を保存
