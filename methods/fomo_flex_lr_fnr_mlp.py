from fomo import FomoClassifier
from fomo.metrics import subgroup_FNR_scorer as metric
from fomo.problem import MLPProblem
from fomo.algorithm import Lexicase

from .train_fomo import train
from sklearn.linear_model import LogisticRegression as ml
from ml.lr import est as base_est

est = FomoClassifier(
    estimator = base_est,
    algorithm = Lexicase(pop_size = 50),
    problem_type = MLPProblem, 
    fairness_metrics=[metric],
    store_final_models=True,
    verbose=False,
    n_jobs=1,
)