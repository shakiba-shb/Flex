import os
from utils import setup_data,evaluate_output,get_hypervolumes, find_bases
import numpy as np
import pandas as pd
import json
from collections import namedtuple
from pathlib import Path

perf = namedtuple('perf',['method','dataset','seed'])

rdirs = ['/home/shakiba/flex/results/Run_16/synthetic1',
            '/home/shakiba/flex/results/Run_16/synthetic1T',
            '/home/shakiba/flex/results/Run_16/synthetic1N',
            '/home/shakiba/flex/results/Run_16/synthetic1W',
            ]

os.makedirs('/home/shakiba/flex/results/Run_16/normalized_hvs_synthetic1', exist_ok=True)

xname = 'subgroup_fnr'
yname = 'accuracy'
base_x = 0
base_y = 0  

#find base values for objectives in one experiment
for rdir in rdirs:
    for f in Path(rdir).glob('*.json'):
        with open(f) as fh:
            perf = json.load(fh)
            base_x_p, base_y_p = find_bases(perf, xname, yname, reverse_y=True)
            if base_x_p > base_x:
                base_x = base_x_p
            if base_y_p > base_y:
                base_y = base_y_p

#get normalized hypervolumes for the experiment
for rdir in rdirs:
    for f in Path(rdir).glob('*.json'):

        with open(f) as fh:
            perf = json.load(fh)
            model_name = perf[0]['method']
            dataset_name = rdir.rsplit('/')[-1]
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
                f'/home/shakiba/flex/results/Run_16/normalized_hvs_adult/hv_{model_name}_{seed}_{dataset_name}.csv',
                index=False
            )
