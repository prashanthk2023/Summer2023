import pandas as pd

train = pd.read_csv(f'./Route_choice_Modelling/Output/train_2019_2_MNL_1.csv')
fare_ls = []
travel_time_ls = []
transfers_ls = []
wait_time_ls = []
pathsize_ls = []
for j in range(1, 8):
    fare_ls.extend(list(train.loc[train[f'avl_{j}'] == 1, f'fare_{j}'].values))
    travel_time_ls.extend(list(train.loc[train[f'avl_{j}'] == 1, f'tt_{j}'].values))
    transfers_ls.extend(list(train.loc[train[f'avl_{j}'] == 1, f'transfers_{j}'].values))
    wait_time_ls.extend(list(train.loc[train[f'avl_{j}'] == 1, f'waiting_time_{j}'].values))
    pathsize_ls.extend(list(train.loc[train[f'avl_{j}'] == 1, f'pathsize_{j}'].values))
print(len(fare_ls))
print(len(travel_time_ls))
print(len(transfers_ls))
print(len(wait_time_ls))
print(len(pathsize_ls))

df = pd.DataFrame({'fare': fare_ls, 'travel_time': travel_time_ls, 'transfers': transfers_ls,'OVTT':wait_time_ls, 'pathsize': pathsize_ls})
coerr = df.corr()

