from amw25.util.parser import parse_input, parse_args
from amw25.util.utils import dumpYAML
from amw25.preprocess.load_data import process_data
from contextlib import redirect_stdout, redirect_stderr
import sys
import pandas as pd
import pickle

def main(argv: list[str] | None = None):
    args = parse_args(argv)
    config = parse_input(args)
    dumpYAML(config, f"{config['dir']['cwd']}/parsed_config.yaml")
    print(config['data']['input'])

    data_dir = config['data']['load']

    pred_args = {
        'X_train': pd.read_csv(f'{data_dir}/X_train_{args.mode}.csv', index_col=0),
        'y_train': pd.read_csv(f'{data_dir}/y_train.csv', index_col=0),
        'X_test': pd.read_csv(f'{data_dir}/X_test_{args.mode}.csv', index_col=0),
        'y_test': pd.read_csv(f'{data_dir}/y_test.csv', index_col=0),
        'X_val': pd.read_csv(f'{data_dir}/X_val_{args.mode}.csv', index_col=0),
        'y_val': pd.read_csv(f'{data_dir}/y_val.csv', index_col=0),
    }

    print('data loaded')

    print('importing xgboost regressor')
    from amw25.models.xgb import XGB_Regressor
    predictor = XGB_Regressor(config, **pred_args)

    with open(f'{config["dir"]["cwd"]}/stdout.x', 'w') as  f, redirect_stdout(f), redirect_stderr(f):
        predictor.main()

if __name__ == '__main__':
    main()


