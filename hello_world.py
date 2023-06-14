import pandas as pd
import warnings

# from tqdm import tqdm

warnings.filterwarnings("ignore")


def checking_all_attributes(df, alt_num):
    attributes = ['waiting_time', 'fare', 'tt', 'pathsize', 'avl']
    alt_ls = list(range(1, alt_num + 1))
    for alt in range(1, alt_num + 1):
        if sum(df[f'avl_{alt}']) != 0:
            for attribute in attributes:
                if sum(df[attribute + f'_{alt}']) == 0:
                    columns_to_drop = [f'waiting_time_{alt}', f'tt_{alt}', f'fare_{alt}', f'pathsize_{alt}',
                                       f'avl_{alt}', f'transfers_{alt}']
                    df.drop(columns_to_drop, axis=1, inplace=True)
                    print(f"Dropped_alt :{alt}")
                    alt_ls.remove(alt)
                    break
        else:
            columns_to_drop = [f'waiting_time_{alt}', f'tt_{alt}', f'fare_{alt}', f'pathsize_{alt}', f'avl_{alt}']
            df.drop(columns_to_drop, axis=1, inplace=True)
            alt_ls.remove(alt)
            print("Dropped_alt :", alt)
    return df, alt_ls


files = ['train_2019_1_MNL.csv', 'test_2019_1_MNL.csv', 'train_2019_2_MNL.csv', 'test_2019_2_MNL.csv',
         'train_2019_3_MNL.csv', 'test_2019_3_MNL.csv', 'train_2019_4_MNL.csv', 'test_2019_4_MNL.csv',
         'train_2019_5_MNL.csv', 'test_2019_5_MNL.csv', 'train_2019_6_MNL.csv', 'test_2019_6_MNL.csv']
# save_files = ['train1_2017_3_MNL.csv', 'train1_2017_4_MNL.csv', 'train1_2017_5_MNL.csv', 'train1_2017_6_MNL.csv']
alt_num = [21, 21, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
for file in range(len(files)):
    train = pd.read_csv(f'./Route_choice_Modelling/Output/{files[file]}')
    print(f'Name of the file: {files[file]}')
    print(f'No of alternatives: {alt_num[file]}')
    print(f'Size before cleaning :{len(train)}')
    items = [f"waiting_time_{x}" for x in range(1, 8)]
    # Filtering the data if waiting time is less than 0 or more than 120 .
    train = train[train.filter(items=items).apply(lambda x: (x >= 0) & (x < 180)).all(axis=1)]
    print(f'Size after cleaning :{len(train)}')
    # Dropping the alternatives if not available or available but the attribute column is having zeroes
    train, alt_rem = checking_all_attributes(train, alt_num[file])
    print(alt_rem)
    att_columns = [f'waiting_time_{alt}' for alt in alt_rem]
    att_columns.extend([f'avl_{alt}' for alt in alt_rem])
    att_columns.extend([f'tt_{alt}' for alt in alt_rem])
    att_columns.extend([f'fare_{alt}' for alt in alt_rem])
    att_columns.extend([f'pathsize_{alt}' for alt in alt_rem])
    att_columns.extend([f'transfers_{alt}' for alt in alt_rem])
    att_columns.extend([f'overlap_percent_{alt}' for alt in alt_rem])
    att_columns.append('choice')
    train = train.loc[:, att_columns]
    train.to_csv(f'./Route_choice_Modelling/Output/{files[file][:-4]}_1.csv', index_label='id')
