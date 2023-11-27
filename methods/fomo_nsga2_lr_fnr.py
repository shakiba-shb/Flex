from fomo import FomoClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2 as algorithm
from fomo.metrics import subgroup_FNR_scorer as metric

from .train_fomo import train
from sklearn.linear_model import LogisticRegression as ml
from ml.lr import est as base_est


est = FomoClassifier(
    estimator = base_est,
    algorithm = algorithm(pop_size=50),
    fairness_metrics=[metric],
    store_final_models=True,
    verbose=True,
    n_jobs=1,
)
