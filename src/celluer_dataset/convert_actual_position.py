import pandas as pd
# from correctIdx import x
from first_dataset import position

data_type = "test"

data_path = "./innoData_{}.csv".format(data_type)
# data_path = "./original_all_towers.csv"
df = pd.read_csv(data_path)

df = df[df["47652"] != 0]
df = df[df["47656"] != 0]
df = df[df["47651"] != 0]
# df = df[df["47653"] != 0]
# df = df[df['47657'] != 0]

print(len(df))

df = df.groupby(
    ["47652", "47656", "47651", "47653", '47657', "point"], as_index=False
).max()

cols = [col for col in df.columns if col != 'point'] + ['point']
df = df[cols]

print(len(df))

# target_ares = {id:loc for id, loc in x.items()}

df = df[df["point"].isin(position.keys())]
df[["x", "y"]] = df["point"].map(position).apply(pd.Series)

print(df.head())

df.to_csv("./{}_dataset.csv".format(data_type), index=None)
# df.to_csv("./original_dataset.csv".format(data_type), index=None)
