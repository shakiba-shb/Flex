from fomo import FomoClassifier
from fomo.metrics import subgroup_FNR_scorer as metric
from fomo.problem import MLPProblem
from fomo.algorithm import Lexicase_NSGA2

from .train_fomo import train
from sklearn.linear_model import LogisticRegression as ml
from ml.lr import est as base_est

est = FomoClassifier(
    estimator = base_est,
    algorithm = Lexicase_NSGA2(pop_size=100),
    problem_type = MLPProblem, 
    fairness_metrics=[metric],
    store_final_models=True,
    verbose=True,
    n_jobs=1,
)