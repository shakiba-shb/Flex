import numpy as np
import warnings
# Data processing
from sklearn.metrics import (accuracy_score, log_loss, precision_score, 
        recall_score, average_precision_score, precision_recall_curve, auc, 
        roc_auc_score)
from deap.tools import hypervolume
from metrics import *
#from fomo.metrics import subgroup_FNR, subgroup_FPR
from clean import clean_dataset
from sklearn.model_selection import train_test_split
from read_file import read_file
#import gerryfair 
import pdb
from itertools import chain
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSEmployment, ACSTravelTime

figsize=(12,4)
from fomo.metrics import subgroup_FPR_loss, subgroup_FNR_loss

def setup_data(dataset,  seed, attributes=None):
    ACSdata = {
    'ACSIncome': ACSIncome,
    'ACSPublicCoverage': ACSPublicCoverage,
    'ACSMobility': ACSMobility,
    'ACSEmployment': ACSEmployment,
    'ACSTravelTime': ACSTravelTime
}

    if attributes is None:
        attributes = dataset.replace('.csv','_protected.csv')
    
    print('setting up data...') 
    dataname = dataset.split('/')[-1].split('.')[0]
    print('dataset:',dataname)
    print('setting up data...') 
    if (dataname == 'mimic4'):
        FEATURES,LABEL,ENCODINGS=read_file(
            dataset,
            label='y',
            text_features=['chiefcomplaint'],
            one_hot_encode=[True],
            ohc_min_frac = 0.01
        )
        ALL_GROUPS = ['gender','ethnicity']
        X_prime = FEATURES[ALL_GROUPS]
        X_train, X_test, X_prime_train, X_prime_test, y_train, y_test = \
            train_test_split(
                FEATURES,
                X_prime,
                LABEL,
                train_size=0.1,
                test_size=0.5,
                random_state=seed,
                stratify=LABEL
            )
        sens_cols=ALL_GROUPS

    elif (dataname in ACSdata.keys()):
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        acs_data = data_source.get_data(states=["MI"], download=False)
        features, label, group = ACSdata[dataname].df_to_pandas(acs_data)
        if ('AGEP' in features.columns):
            bins=[0, .25, .50, .75, 1] 
            features['AGEP'] = pd.qcut(features['AGEP'], q=bins, labels=False, duplicates='drop',retbins=False)
            group['AGEP'] = pd.qcut(group['AGEP'], q=bins, labels=False, duplicates='drop',retbins=False)
        print(features['AGEP'].value_counts())
        print(group['AGEP'].value_counts())

        print("#samples = ", len(features))
        label = label.squeeze()
        X_train, X_test, X_prime_train, X_prime_test, y_train, y_test,  = train_test_split(
            features, group, label, train_size=0.3, test_size=0.5, random_state=seed)
        sens_cols=group.columns
        print('sensitive features: {}'.format(sens_cols))
    else:
        X, X_prime, y = clean_dataset(dataset, attributes, centered=True)
        X_train, X_test, X_prime_train, X_prime_test, y_train, y_test = \
            train_test_split(
                X,
                X_prime,
                y,
                test_size=0.5,
                random_state=seed,
                stratify=y
            )
        sens_df = pd.read_csv(attributes)
        sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    
    print('positive labels in test:',np.sum(y_test==1))
      
    return X_train, X_test, X_prime_train, X_prime_test, y_train, y_test, sens_cols

def evaluate_output(X, X_prime, y, predictions, probabilities):
    """returns metrics for comparison for a single model"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # auditor_fp = gerryfair.model.Auditor(X_prime, y.values, 'FP')
        # auditor_fn = gerryfair.model.Auditor(X_prime, y.values, 'FN')
        # _,auditor_fp_violation = auditor_fp.audit(predictions)
        # _,auditor_fn_violation = auditor_fn.audit(predictions)

    # requires estimator
    sg_fpr = subgroup_FPR_loss(y.values, probabilities, X_prime)
    sg_fnr = subgroup_FNR_loss(y.values, probabilities, X_prime)

    accuracy = accuracy_score(y,predictions) 
    fpr = np.mean(false_positives(y,predictions))
    logloss = log_loss(y, probabilities) 
    mae = MAE(y, probabilities) 
    if np.array(predictions==0).all():
        prec, recall, aps, auc_prc, auc_roc = 0, 0, 0, 0, 0
    else:
        prec = precision_score(y, predictions) 
        recall = recall_score(y, predictions) 
        aps = average_precision_score(y, probabilities)
        pc, rc, _ = precision_recall_curve(y, probabilities)
        auc_prc = auc(rc,pc)
        auc_roc = roc_auc_score(y,probabilities)
    
               
    scores = {
            # 'auditor_fp_violation':auditor_fp_violation,
            # 'auditor_fn_violation':auditor_fn_violation,
            'subgroup_fpr':sg_fpr,
            'subgroup_fnr':sg_fnr,
            'accuracy':accuracy,
            'fpr':fpr,
            'logloss':logloss,
            'mae':float(mae),
            'precision':prec,
            'recall':recall,
            'ave_precision_score':aps,
            'auc_prc':auc_prc,
            'auc_roc':auc_roc
           } 
    
    return scores 

# PARETO FRONT TOOLS
def check_dominance(p1,p2):

    flag1 = 0
    flag2 = 0

    for o1,o2 in zip(p1,p2):
        if o1 < o2:
            flag1 = 1
        elif o1 > o2:
            flag2 = 1

    if flag1==1 and flag2 == 0:
        return 1
    elif flag1==0 and flag2 == 1:
        return -1
    else:
        return 0

def front(obj1,obj2):
    """return indices from x and y that are on the Pareto front."""
    rank = []
    assert(len(obj1)==len(obj2))
    n_inds = len(obj1)
    front = []

    for i in np.arange(n_inds):
        p = (obj1[i],obj2[i])
        dcount = 0
        dom = []
        for j in np.arange(n_inds):
            q = (obj1[j],obj2[j])
            compare = check_dominance(p,q)
            if compare == 1:
                dom.append(j)
#                 print(p,'dominates',q)
            elif compare == -1:
                dcount = dcount +1
#                 print(p,'dominated by',q)

        if dcount == 0:
#             print(p,'is on the front')
            front.append(i)

#     f_obj1 = [obj1[f] for f in front]
    f_obj2 = [obj2[f] for f in front]
#     s1 = np.argsort(np.array(f_obj1))
    s2 = np.argsort(np.array(f_obj2))
#     front = [front[s] for s in s1]
    front = [front[s] for s in s2]

    return front
# metrics
fair_metrics = {
    # 'auditor_fp_violation':'Audit FP Violation $\gamma$',
    # 'auditor_fn_violation':'Audit FN Violation $\gamma$',
    'subgroup_fpr': 'Subgroup FPR',
    'subgroup_fnr': 'Subgroup FNR',
    }
loss_metrics = {
    'accuracy':'1-Accuracy',
    'ave_precision_score':'1-Average Precision Score',
    'auc_prc':'1 - Area Under Precision-Recall Curve',
    'auc_roc':'1 - AUROC'
    }
reverse_metrics = ['accuracy','precision','recall','ave_precision_score','auc_prc','auc_roc']
method_nice = {
    'gerryfair':'GerryFair',
    'gerryfair_xgb':'GerryFairGB',
    'feat_lex':'LEX',
    'feat_tourn':'Tourn',
    'feat_random_p100_g0':'Random0',
    'feat_random_p100_g100':'Random100',
    # 'feat_flex':'Flex',
    'feat_flex2':'FLEX',
    'feat_nsga2':'NSGA2',
    # 'feat_flex_nsga2':'FLEX-NSGA2',
    'feat_flex2_nsga2':'FLEX-NSGA2',
}
# Hypervolume tools
from deap.tools._hypervolume import pyhv 
# compute hypervolumes of the Pareto front
            
def get_hypervolume(perf, xname, yname, reverse_x=False, 
		    reverse_y = False):
    base_lines = {
    'adult': {'base_auroc': 0.1629644152534065, 'base_subgroup_fnr': 0.0080437901922114},
    'student': {'base_auroc': 0, 'base_subgroup_fnr': 0},
    'lawschool': {'base_auroc': 0, 'base_subgroup_fnr': 0},
    'mimic4': {'base_auroc': 0, 'base_subgroup_fnr': 0},
    'ACSIncome': {'base_auroc': 0, 'base_subgroup_fnr': 0}
    }
   
    x_vals = {'train':[], 'test':[]}
    y_vals = {'train':[], 'test':[]}
    
    for i,p in enumerate(perf):
        for t in ['train','test']:
            x_vals[t].append(p[t][xname])
            y_vals[t].append(p[t][yname])
    dataset = perf[0]['dataset']    
    if reverse_x: 
        for t in ['train','test']:
            x_vals[t] = [-x for x in x_vals[t]]
    if reverse_y: 
        for t in ['train','test']:
            y_vals[t] = [1-y for y in y_vals[t]]
    for t in ['train','test']:
        x_vals[t] = np.array(x_vals[t])
        y_vals[t] = np.array(y_vals[t])
        
    metric = 'hv('+xname+':'+yname+')'
    hv = {'train':{}, 'test':{}}
    for t in ['train','test']:
        PF = front(x_vals[t],y_vals[t])
        pf_x = [x_vals[t][i]/base_lines[dataset]['base_subgroup_fnr'] for i in PF]
        pf_y = [y_vals[t][i]/base_lines[dataset]['base_auroc'] for i in PF]
        hv[t] = pyhv.hypervolume([(xi,yi) for xi,yi in zip(pf_x,pf_y)],
                                                  ref=np.array([1,1]))
    return [{'train':True, metric:hv['train']},
            {'train':False, metric:hv['test']}]

def get_hypervolumes(perf):
    hv = [] 
    for f in fair_metrics.keys():
        for L in loss_metrics.keys():
            hv += get_hypervolume(perf, f, L, reverse_y = L in reverse_metrics)
                     
    return hv

# Pareto front plots
import matplotlib.pyplot as plt
def pareto_plot(perf,dataset_name,xname,yname,xname_nice,yname_nice,
               reverse_x = False, reverse_y = False):
    h = plt.figure(figsize=figsize)
    x_vals = {'train':[], 'test':[]}
    y_vals = {'train':[], 'test':[]}
    
    for i,p in enumerate(perf):
        for t in ['train','test']:
            x_vals[t].append(p[t][xname])
            y_vals[t].append(p[t][yname])
        
    if reverse_x: 
        for t in ['train','test']:
            x_vals[t] = [-x for x in x_vals[t]]
    if reverse_y: 
        for t in ['train','test']:
            y_vals[t] = [1-y for y in y_vals[t]]
    for t in ['train','test']:
        x_vals[t] = np.array(x_vals[t])
        y_vals[t] = np.array(y_vals[t])
    s = np.argsort(y_vals['train'])
    plt.plot(x_vals['train'][s],y_vals['train'][s], '--b',marker='.',label='train')
    plt.plot(x_vals['test'][s],y_vals['test'][s], '--r',marker='x', label='test')
    for i in np.arange(len(x_vals['train'])):
        plt.plot([x_vals[t][i] for t in ['train','test']],
                 [y_vals[t][i] for t in ['train','test']],
                 ':',alpha=0.5,label='_no_legend')
    plt.legend()

    plt.title(dataset_name)
    plt.xlabel(xname_nice)
    plt.ylabel(yname_nice)
    return h

def pareto_compare_plot(perf1,perf2,dataset_name,xname,yname,xname_nice,
        yname_nice, reverse_x = False, reverse_y = False, rdir=''):
    h = plt.figure(figsize=figsize)
    
    model1 = perf1[0]['method']
    model2 = perf2[0]['method']
    
    seed = perf1[0]['seed']
    assert seed == perf2[0]['seed'] 
    seed = str(seed)
    
    x_vals = {model1:{'train':[], 'test':[]},
              model2:{'train':[], 'test':[]}}
    y_vals = {model1:{'train':[], 'test':[]},
              model2:{'train':[], 'test':[]}}
    
    for perf,m in zip([perf1,perf2],[model1,model2]):
        for i,p in enumerate(perf):
            for t in ['train','test']:
                x_vals[m][t].append(p[t][xname])
                y_vals[m][t].append(p[t][yname])
        
    if reverse_x: 
        # print('reversing x for ',xname)
        for m in [model1,model2]:
            for t in ['train','test']:
                x_vals[m][t] = [-x for x in x_vals[m][t]]
    if reverse_y: 
        # print('reversing y for ',yname)
        for m in [model1,model2]:
            for t in ['train','test']:
                y_vals[m][t] = [1-y for y in y_vals[m][t]]
            
    # get the pareto front of the combined solutions
    all_x_vals_test = x_vals[model1]['test'] + x_vals[model2]['test']
    all_y_vals_test = y_vals[model1]['test'] + y_vals[model2]['test']
#     print('all_x_vals_test:',all_x_vals_test)
#     print('all_y_vals_test:',all_y_vals_test)
    PF = front(all_x_vals_test,all_y_vals_test)
    pf_test_x_vals = [all_x_vals_test[i] for i in PF]
    pf_test_y_vals = [all_y_vals_test[i] for i in PF]
    
    for m in [model1,model2]:
        for t in ['train','test']:
            x_vals[m][t] = np.array(x_vals[m][t])
            y_vals[m][t] = np.array(y_vals[m][t])
            
    pf_x = {}
    pf_y = {}
    for m in [model1,model2]:
        pf_x[m] = {'train':[],'test':[]}
        pf_y[m] = {'train':[],'test':[]}
        for t in ['train','test']:
            pf_tmp = front(x_vals[m][t],y_vals[m][t])
            pf_x[m][t] = [x_vals[m][t][i] for i in pf_tmp]
            pf_y[m][t] = [y_vals[m][t][i] for i in pf_tmp]
            
    # make plots!
        
    plt.plot(pf_test_x_vals, pf_test_y_vals, '-k', linewidth=2,alpha=0.2, 
            label= '_no_legend')

    for m,c, in zip([model1,model2],['r','b']):
#         plt.scatter(x_vals[m]['train'],y_vals[m]['train'],color=c,marker='.',label=m+' train',
#                    alpha=0.2)
        plt.plot(pf_x[m]['train'],pf_y[m]['train'],':.',color=c,label=m+' train',#label='_no_legend',
                   alpha=0.3)
#         plt.scatter(x_vals[m]['test'],y_vals[m]['test'], color=c,marker='x', label=m+' test')
        plt.plot(pf_x[m]['test'],pf_y[m]['test'],'--x',color=c,label=m+' test', #label='_no_legend',
                   alpha=0.5)
   
#     print('PF:',PF)
    plt.scatter(pf_test_x_vals, pf_test_y_vals, marker='o', facecolor='', edgecolor='k', 
            s=200, label= 'Pareto front',alpha=0.5)
#     for i in np.arange(len(x_vals['train'])):
#         plt.plot([x_vals[t][i] for t in ['train','test']],
#                  [y_vals[t][i] for t in ['train','test']],
#                  ':',alpha=0.5,label='_no_legend')
    leg = plt.legend(loc=[1.01, 0.5])

    plt.title(dataset_name)
    plt.xlabel(xname_nice)
    plt.ylabel(yname_nice)

    # plt.tight_layout()
    if not os.path.exists('../paper/figs/pareto/'+rdir):
        os.mkdir('../paper/figs/pareto/'+rdir)
    savename = ('../paper/figs/pareto/'+rdir
            +'/pareto_compare_'+xname+'-'+yname+'_'
            +'_'+model1+'-'+model2
            +'_'+dataset_name+'_'+seed+'.png')
    print('saving ',savename)
    plt.savefig(savename, bbox_extras=[leg], dpi=400)
    
    # return hypervolumes 
    return h

def pareto_multicompare_plot(perfs,dataset_name,xname,yname,xname_nice,
        yname_nice, reverse_x = False, reverse_y = False, rdir=''):
    h = plt.figure(figsize=figsize)
    plt.gca().grid(True,axis='both')
    
    models = [p[0]['method'] for p in perfs]
    # sort results by name
    s = np.argsort([method_nice[m] for m in models])
    models = [models[i] for i in s]
    perfs = [perfs[i] for i in s]
    
    seed = perfs[0][0]['seed']
    for p in perfs:
        assert seed == p[0]['seed'] 
    seed = str(seed)
    
    x_vals = {m:{'train':[], 'test':[]} for m in models}
    y_vals = {m:{'train':[], 'test':[]} for m in models}
    
    for perf,m in zip(perfs,models):
        for i,p in enumerate(perf):
            for t in ['train','test']:
                x_vals[m][t].append(p[t][xname])
                y_vals[m][t].append(p[t][yname])
        
    if reverse_x: 
        # print('reversing x for ',xname)
        for m in models:
            for t in ['train','test']:
                x_vals[m][t] = [-x for x in x_vals[m][t]]
    if reverse_y: 
        # print('reversing y for ',yname)
        for m in models:
            for t in ['train','test']:
                y_vals[m][t] = [1-y for y in y_vals[m][t]]
            
    # get the pareto front of the combined solutions
    all_x_vals_test = list(chain.from_iterable(
        x_vals[m]['test'] for m in models))
    all_y_vals_test = list(chain.from_iterable(
        y_vals[m]['test'] for m in models))
#     print('all_x_vals_test:',all_x_vals_test)
#     print('all_y_vals_test:',all_y_vals_test)
    PF = front(all_x_vals_test,all_y_vals_test)
    pf_test_x_vals = [all_x_vals_test[i] for i in PF]
    pf_test_y_vals = [all_y_vals_test[i] for i in PF]
    
    for m in models:
        for t in ['train','test']:
            x_vals[m][t] = np.array(x_vals[m][t])
            y_vals[m][t] = np.array(y_vals[m][t])
            
    pf_x = {}
    pf_y = {}
    for m in models:
        pf_x[m] = {'train':[],'test':[]}
        pf_y[m] = {'train':[],'test':[]}
        for t in ['train','test']:
            pf_tmp = front(x_vals[m][t],y_vals[m][t])
            pf_x[m][t] = [x_vals[m][t][i] for i in pf_tmp]
            pf_y[m][t] = [y_vals[m][t][i] for i in pf_tmp]
            
    # make plots!
        
    plt.plot(pf_test_x_vals, pf_test_y_vals, '-k', linewidth=4,alpha=0.1, 
            label= '_no_legend')
    cmap = plt.cm.get_cmap('tab20',len(models))
    markers = ['x','s','^','d','*','+','v','>','<','D']
    # pdb.set_trace()
    for c,m in enumerate(models):
#         plt.scatter(x_vals[m]['train'],y_vals[m]['train'],color=c,marker='.',label=m+' train',
#                    alpha=0.2)
        # plt.plot(pf_x[m]['train'],pf_y[m]['train'],':.',color=cmap(c),
        #         label=m+' train',#label='_no_legend',
        #         alpha=0.3)
#         plt.scatter(x_vals[m]['test'],y_vals[m]['test'], color=c,marker='x', label=m+' test')
        plt.plot(pf_x[m]['test'],pf_y[m]['test'],markers[c],
                color=cmap(c),#markerfacecolor='none',
                label=method_nice[m], #label='_no_legend',
                   alpha=1.0)
        plt.plot(pf_x[m]['test'],pf_y[m]['test'],'--',color=cmap(c),
                label='_no_legend',
                   alpha=0.7)
   
#     print('PF:',PF)
    plt.scatter(pf_test_x_vals, pf_test_y_vals, marker='o', facecolor='', 
            edgecolor='k', s=200, label= 'Pareto front',alpha=0.5)
#     for i in np.arange(len(x_vals['train'])):
#         plt.plot([x_vals[t][i] for t in ['train','test']],
#                  [y_vals[t][i] for t in ['train','test']],
#                  ':',alpha=0.5,label='_no_legend')
    leg = plt.legend()#loc=[1.01, 0.5])

    plt.title(dataset_name)
    plt.xlabel(xname_nice)
    plt.ylabel(yname_nice)

    # plt.tight_layout()
    if not os.path.exists('../paper/figs/pareto/'+rdir):
        os.mkdir('../paper/figs/pareto/'+rdir)
    savename = ('../paper/figs/pareto/'+rdir
            +'/pareto_multicompare_'+xname+'-'+yname+'_'
            +'_'+'-'.join([m for m in models])
            +'_'+dataset_name+'_'+seed+'.png')
    print('saving ',savename)
    plt.savefig(savename, bbox_extras=[leg], dpi=400)
    
    # return hypervolumes 
    return h

def pareto_plots(perf, dataset_name):
    for f,flabel in fair_metrics.items():
        for L,Llabel in loss_metrics.items():
            pareto_plot(perf, dataset_name, f, L, flabel, Llabel)

def pareto_compare_plots(perf1,perf2, dataset_name):
    for f,flabel in fair_metrics.items():
        for L,Llabel in loss_metrics.items():
            pareto_compare_plot(perf1,perf2, dataset_name, f, L, flabel, Llabel,
                               reverse_y = L in reverse_metrics)

def pareto_multicompare_plots(perfs, dataset_name, fm = fair_metrics,
        lm = loss_metrics, rdir=''):
    for f,flabel in fm.items():
        for L,Llabel in lm.items():
            pareto_multicompare_plot(perfs, dataset_name, f, L, flabel, Llabel,
                               reverse_y = L in reverse_metrics, rdir=rdir)
