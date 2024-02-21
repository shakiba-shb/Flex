"""Evaluate a method on a dataset"""
import ipdb
import copy
import os
from utils import setup_data,evaluate_output,get_hypervolumes, find_bases
import numpy as np
import pandas as pd
import time
import json
import importlib
import warnings
from pymoo.indicators.hv import Hypervolume
from collections import namedtuple
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

perf = namedtuple('perf',['method','dataset','seed'])
rdir = 'results/Run_15/synthetic1W'
os.makedirs(f'{rdir}/normalized_hvs', exist_ok=True)

xname = 'subgroup_fnr'
yname = 'auc_roc'
base_x = 0
base_y = 0  

#find base values for objectives in one experiment
for f in Path(rdir).glob('*.json'):
    print(f)
    with open(f) as fh:
        perf = json.load(fh)
        base_x_p, base_y_p = find_bases(perf, xname, yname, reverse_y=True)
        if base_x_p > base_x:
            base_x = base_x_p
        if base_y_p > base_y:
            base_y = base_y_p


#get normalized hypervolumes for the experiment
for f in Path(rdir).glob('*.json'):

    with open(f) as fh:
        perf = json.load(fh)
        model_name = perf[0]['method']
        dataset_name = perf[0]['dataset']
        seed = perf[0]['seed']

        header = {
        'method':model_name,
        'dataset':dataset_name,
        'seed':seed
        }
        hv = get_hypervolumes(perf, base_x, base_y)
        hv = [{**header, **i} for i in hv]
        df_hv = pd.DataFrame.from_records(hv)
        df_hv.to_csv(
            f'{rdir}/normalized_hvs/hv_{model_name}_{seed}_{dataset_name}.csv',
            index=False
        )
