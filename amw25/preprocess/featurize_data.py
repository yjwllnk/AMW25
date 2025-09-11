from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition.element import TMetalFraction, BandCenter
from matminer.featurizers.composition.orbital import ValenceOrbital

import pandas as pd

SYM_COL = ['n_rot','n_inv','n_rotoinv', 'n_mirror', 'n_rot_screw','1/2tran_screw','1/3tran_screw','3/4tran_screw','1/4tran_screw','1/6tran_screw','n_glide','1/2tran_glide','3/4tran_glide','1/4tran_glide','n_tr']

def preprocess_data(config):
    data_dir = config['data']['save']
    conf = config['data']

    df = pd.read_csv(conf['input'])
    label_list = [f"{df['formula'].values[i]}_{df['temperature'].values[i]}" for i in range(len(df))]
    df['label'] = label_list
    df_clean = df[['formula','label', 'sg', 'temperature', 'seebeck', 'cond_elec', 'cond_therm'] + SYM_COL]
    df_clean.to_csv(f"{data_dir}/df_clean.csv", index=False)

def featurize_data(config):
    data_dir = config['data']['save']
    conf = config['data']

    df_clean = pd.read_csv(f"{data_dir}/df_clean.csv")
    
    # Convert string to composition
    str_to_comp = StrToComposition()
    df = str_to_comp.featurize_dataframe(df_clean, 'formula', ignore_errors=True)

    # Featurize composition
    element_property = ElementProperty.from_preset(preset_name="magpie")
    df = element_property.featurize_dataframe(df, 'composition', ignore_errors=True)

    # Add additional features
    df = TMetalFraction().featurize_dataframe(df, 'composition', ignore_errors=True)
    df = BandCenter().featurize_dataframe(df, 'composition', ignore_errors=True)
    df = ValenceOrbital().featurize_dataframe(df, 'composition', ignore_errors=True)
    df.dropna(inplace=True, axis=1)

    df.to_csv(f"{data_dir}/df_featureized.csv", index=False)


def main(argv: list[str] | None = None):
    import yaml
    from amw25.parser import parse_input, parse_args

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    args = parse_args(argv)
    config = parse_input(args)
    preprocess_data(config)
    featurize_data(config)

    print("Data preprocessing and feature extraction completed.")


if __name__ == '__main__':
    main()
