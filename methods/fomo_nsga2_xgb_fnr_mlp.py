from fomo import FomoClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2 as algorithm
from fomo.metrics import subgroup_FNR as metric
from fomo.problem import MLPProblem

from .train_fomo import train
from xgboost.sklearn import XGBClassifier as ml
from ml.rf import est as base_est

est = FomoClassifier(
    estimator = base_est,
    algorithm = algorithm(pop_size=100),
    problem_type = MLPProblem, 
    fairness_metrics=[metric],
    store_final_models=True,
    verbose=True,
    n_jobs=1,
)
