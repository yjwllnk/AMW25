import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import re

import yaml

def plot_parity(y_test, y_pred, r2, config, mae, rmse):
   fig, axs = plt.subplots(1, 3, figsize=(15, 6), dpi=300)
   plt.suptitle(f"{config['model']['type']}-{config['data']['input']} test")
   for i, ax in enumerate(axs):
      label = config['data']['target'][i]
      ax.set_box_aspect(1)
      ax.axline(xy1=(0,0), slope=1, color="gray", linewidth=1, linestyle="dashed")
      ax.scatter(y_test[i], y_pred[i], marker='o', color='#6699FF', s=40, edgecolor='black', label=f'R2: {r2[i]:.3f}\nMAE:{mae[i]:.3f}\nRMSE:{rmse[i]:.3f}')
      ax_lim =[int(min(y_test[i].min(), y_pred[i].min())) - 0.5, int(max(y_test[i].max(), y_pred[i].max())) + 0.5]
      ax.set_xlim(ax_lim)
      ax.set_ylim(ax_lim)
      ax.set_xlabel(f"Reference label")
      ax.set_ylabel(f"Predicted label")
      ax.tick_params(axis='both',which='major',length=4,labelsize=12,width=1)
      ax.legend(fontsize=12, loc='upper left')
   plt.savefig(f"{config['plot']['save']}/{config['data']['input']}_parity.png")
   plt.close()

def dict_representer(dumper, data=None):
    return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data, flow_style=False)
def list_representer(dumper, data=None):
    return dumper.represent_sequence(yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, data, flow_style=True)

class WDumper(yaml.Dumper):
    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()

def dumpYAML(data, filename, indent=4, sort_keys=False, explicit_start=True, explicit_end=True, default_flow_style=False,):
    yaml.add_representer(dict, dict_representer, Dumper=WDumper)
    yaml.add_representer(list, list_representer, Dumper=WDumper)
    with open(filename, 'w') as fp:
        yaml.dump(data, fp, Dumper=WDumper, sort_keys=sort_keys, explicit_start=explicit_start, explicit_end=explicit_end, default_flow_style=default_flow_style, indent=indent)



