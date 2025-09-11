import argparse
import os
import yaml

def parse_args(argv: list[str] | None=None):
    parser = argparse.ArgumentParser(description = 'cli tool for quick & dirty ML experiments')

    parser.add_argument('--cwd', type=str, default='test',
                        help='directory to save data & results; will be created in cwd')

    parser.add_argument('--config',type=str, default='config.yaml',
                        help='file with parameters to be used; default configuration at ./examples')

    parser.add_argument('--model',type=str, default='xgb',
                        help='ML model type; options: linear_regressor, xgb, mlp')

    parser.add_argument('--mode',type=str, default='tot',
                        help='input data type; options: sym, mat, both')

    parser.add_argument('--load',type=str, default='/home/jinvk/AMW25/data',
                        help='input data type; options: sym, mat, both')

    parser.add_argument('--scale_X',type=bool, default=False,
                        help='input data type; options: sym, mat, both')

    parser.add_argument('--scale_y',type=bool, default=True,
                        help='input data type; options: sym, mat, both')

    parser.add_argument('--stop',type=float, default=0.75,
                        help='input data type; options: sym, mat, both')

    return parser.parse_args(argv)

def parse_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def check_config(config):
    # assert os.path.exists(config['data']['input'])
    assert config['model']['type'] in ['linear_regressor', 'xgb', 'mlp']
    assert config['data']['input'] in ['sym', 'mat', 'both', 'tot']

def parse_input(args):
    config = parse_config(args.config)
    config['dir']['prefix'] =args.cwd
    config['dir']['cwd'] =os.path.join(os.path.abspath(os.getcwd()), args.cwd)
    config['model']['type'] = args.model
    config['data']['save'] = args.load
    config['data']['input'] = args.mode
    config['data']['scale_X'] = args.scale_X
    config['data']['scale_y'] = args.scale_y
    config['data']['load'] = args.load
    config['model']['optuna']['stop'] = args.stop

    keys = ['plot', 'model']
    for key in keys:
        config[key]['save'] = os.path.join(config['dir']['cwd'], config[key]['save'])
        os.makedirs(config[key]['save'], exist_ok=True)
    check_config(config)

    return config
