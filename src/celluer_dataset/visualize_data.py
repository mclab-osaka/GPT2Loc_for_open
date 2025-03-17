import matplotlib.pyplot as plt
from first_dataset import position
import pandas as pd

location_dict = position

data_type = "test"
data_path = "./innoData_{}.csv".format(data_type)
df_tmp = pd.read_csv(data_path)
pos_list = list(df_tmp["point"].unique())

# 辞書からx座標とy座標を抽出
# ids = list(location_dict.keys())  # 場所のID
ids = pos_list
x_coords = [location_dict[id]["x"] for id in ids]  # x座標
y_coords = [location_dict[id]["y"] for id in ids]  # y座標

# プロット
plt.figure(figsize=(8, 6))
plt.scatter(x_coords, y_coords, color='blue', label='Locations')


x_coords = [coord["x"] for coord in location_dict.values()]  # x座標
y_coords = [coord["y"] for coord in location_dict.values()]  # y座標

plt.scatter(x_coords, y_coords, color='red', label='all Locations', alpha=0.3)


# 各点にIDをラベル付け
# for i, txt in enumerate(ids):
#     plt.text(x_coords[i], y_coords[i], str(txt), fontsize=10, ha='right', va='bottom')

# グラフの設定
plt.title('2D Location Plot', fontsize=14)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.grid(True)
# plt.legend()
plt.tight_layout()

# プロットの表示
plt.savefig("hoge.pdf")