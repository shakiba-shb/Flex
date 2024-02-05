"""Evaluate a method on a dataset"""
import ipdb
import copy
import os
from utils import setup_data,evaluate_output,get_hypervolumes
import numpy as np
import pandas as pd
import time
import json
import importlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

def evaluate(model_name, dataset, seed, rdir):
    """Evaluates the estimator in methods/{model_name}.py on dataset and stores
    results.
    """
    print(f'training {model_name} on {dataset}, seed={seed}, rdir={rdir}')

    os.makedirs(rdir, exist_ok=True)
    # data setup
    X_train, X_test, X_prime_train, X_prime_test, y_train, y_test, sens_cols = \
    setup_data(dataset, seed)
    dataset_name = dataset.split('/')[-1].split('.')[0]
    print(f'X_train size: {X_train.shape}')

    # train algorithm
    alg = importlib.import_module(f"methods.{model_name}")

    t0 = time.process_time()
    res = alg.train(alg.est, X_train, X_prime_train, y_train, X_test, sens_cols)

    train_predictions=res[0]
    test_predictions=res[1]
    train_probabilities=res[2]
    test_probabilities=res[3]
    history = res[4]
    best_est = res[5]

    #Save pareto front for each generation for one seed only
    output_directory = os.path.join(rdir, 'pareto_history')
    os.makedirs(output_directory, exist_ok=True)
    for i, gen in enumerate(history):
        objectives = np.array(gen.opt.get("F")).tolist()
        ests = np.array(gen.opt.get("X")).tolist()
        fngs = np.array(gen.opt.get("fng")).tolist()
        fns = np.array(gen.opt.get("fn")).tolist()
        data = {'objectives': objectives, 'ests': ests, 'fngs': fngs, 'fns': fns}
        file_path = os.path.join(output_directory, f'{model_name}_{seed}_generation_{i+1}.json')
        #save best estimator data in the last generation
        if (i == len(history) - 1):
            data['best_est_F'] = best_est.get("F").tolist()
            data['best_est_X'] = best_est.get("X").tolist()
            data['best_est_fng'] = best_est.get("fng").tolist()
            data['best_est_fn'] = best_est.get("fn").tolist()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    performance = []
    for i, (train_pred, test_pred, train_prob, test_prob) in enumerate(zip(
        train_predictions,
        test_predictions,
        train_probabilities, 
        test_probabilities
        )):
        if len(train_prob.shape) > 1 and train_prob.shape[1] == 2:
            train_prob = train_prob[:,1] 
        if len(test_prob.shape) > 1 and test_prob.shape[1] == 2:
            test_prob = test_prob[:,1] 
        performance.append({
            'method':model_name,
            'model':model_name+':archive('+str(i)+')',
            'dataset':dataset_name,
            'seed':seed,
            'train':evaluate_output(X_train, X_prime_train, y_train, train_pred, 
                train_prob),
            'test':evaluate_output(X_test, X_prime_test, y_test, test_pred, 
                test_prob)
        })
        
    runtime = time.process_time() - t0
    header = {
            'method':model_name,
            'dataset':dataset_name,
            'seed':seed,
            'time':runtime
    }
    # get hypervolume of pareto front
    hv = get_hypervolumes(performance)
    hv = [{**header, **i} for i in hv]
    df_hv = pd.DataFrame.from_records(hv)
    df_hv.to_csv(
            f'{rdir}/hv_{model_name}_{seed}_{dataset_name}.csv',
            index=False
        )
    
    with open(f'{rdir}/perf_{model_name}_{dataset_name}_{seed}.json', 'w') as fp:
        json.dump(performance, fp, sort_keys=True, indent=2)
    return performance, df_hv

import argparse
if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('-data', action='store', type=str, default='data/adult.csv',
                        help='Data file to analyze')
    # parser.add_argument('-atts', action='store', type=str,
    #                     help='File specifying protected attributes')
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', default='fomo_lex_lr_fnr_linear',type=str,
            help='Name of estimator (with matching file in ml/)')
    parser.add_argument('-rdir', action='store', default='results/report', type=str,
                        help='Name of save file')
    parser.add_argument('-seed', action='store', default=42, type=int, help='Seed / trial')
    args = parser.parse_args()

    evaluate( args.ml, args.data, args.seed, args.rdir)