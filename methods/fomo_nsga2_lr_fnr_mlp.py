from fomo import FomoClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2 as algorithm
from fomo.metrics import subgroup_FNR_scorer, subgroup_accuracy_scorer, subgroup_FPR_scorer, subgroup_log_loss_scorer
from fomo.problem import MLPProblem
from sklearn.metrics import make_scorer, log_loss
from .train_fomo import train
from sklearn.linear_model import LogisticRegression as ml
from ml.lr import est as base_est

est = FomoClassifier(
    estimator = base_est,
    algorithm = algorithm(pop_size=100),
    problem_type = MLPProblem, 
    accuracy_metrics=[make_scorer(log_loss, needs_proba=True)],
    fairness_metrics=[subgroup_log_loss_scorer],
    store_final_models=True,
    verbose=True,
    n_jobs=1,
)
