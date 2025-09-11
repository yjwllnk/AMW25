from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import pandas as pd
import pickle

def process_data(config):
    conf = config['data']
    data_dir = conf['load']
    input_type = conf['input'].lower()
    data_og = '/home/jinvk/AMW25/data/raw'

    if input_type == 'sym':
        df = pd.read_csv(f"{data_og}/data_sym.csv")

    elif input_type == 'mat':
        df = pd.read_csv(f"{data_og}/data_mat.csv")

    elif input_type == 'both':
        df = pd.read_csv(f"{data_og}/data_both.csv")

    else:
        raise ValueError(f"Unknown input type: {input_type}")

    train_ratio, test_ratio, val_ratio = conf['split']
    train_ratio = train_ratio / (train_ratio + test_ratio)
    test_ratio = test_ratio / (train_ratio + test_ratio)

    target_col = conf['target']
    drop_col = conf['drop']

    df_train, df_val = train_test_split(df, test_size=val_ratio, random_state=conf['random_seed'])

    df_train_X = df_train.drop(columns = target_col + drop_col)
    df_train_y = df_train[target_col]
    df_train_X.to_csv(f'{data_dir}/df_train_X.csv')
    df_train_y.to_csv(f'{data_dir}/df_train_y.csv')

    df_val_X = df_val.drop(columns = target_col + drop_col)
    df_val_y = df_val[target_col]
    df_val_X.to_csv(f'{data_dir}/df_val_X.csv')
    df_val_y.to_csv(f'{data_dir}/df_val_y.csv')

    X_train, X_test, y_train, y_test = train_test_split(df_train_X, df_train_y, test_size=test_ratio, random_state=conf['random_seed'])
    X_val = df_val_X

    if conf['scale_X']:
        x_scaler = StandardScaler()
        x_scaler.fit(df_train_X)
        X_train = x_scaler.transform(X_train)
        X_test = x_scaler.transform(X_test)
        X_val = x_scaler.transform(df_val_X)

    if conf['scale_y']:
        y_scaler = StandardScaler()
        y_scaler.fit(df_train_y)
        y_train = y_scaler.transform(y_train)
        y_test = y_scaler.transform(y_test)
        y_val = y_scaler.transform(df_val_y)

    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,

        # 'x_scaler': x_scaler,
        # 'X_train_raw': X_train,
        # 'X_test_raw': X_test,
        # 'X_val': X_val,
        # 'y_val': y_val,
        # 'X_val_raw': X_val,
        # 'target_col': target_col,
        # 'drop_col': drop_col,
        # 'input_type': input_type,
        # 'train_ratio': train_ratio,
        # 'test_ratio': test_ratio,
        # 'val_ratio': val_ratio,
    }

    with open(f"{data_dir}/data_{conf['input']}.pkl", 'wb') as f:
        pickle.dump(data_dict, f)
        print(data_dict)
    return data_dict

def main(config):
    process_data(config)

if __name__ == "__main__":
    import yaml
    import sys
    with open(sys.argv[1], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
