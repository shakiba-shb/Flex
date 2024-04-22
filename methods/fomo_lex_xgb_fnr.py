from fomo import FomoClassifier
from fomo.metrics import subgroup_FNR_scorer, subgroup_accuracy_scorer, subgroup_FPR_scorer, subgroup_log_loss_scorer
from fomo.algorithm import Lexicase
from sklearn.metrics import make_scorer, log_loss
from .train_fomo import train
from xgboost.sklearn import XGBClassifier as ml
from ml.rf import est as base_est


est = FomoClassifier(
    estimator = base_est,
    algorithm = Lexicase(pop_size=100),
    accuracy_metrics=[make_scorer(log_loss, needs_proba=True)],
    fairness_metrics=[subgroup_log_loss_scorer],
    store_final_models=True,
    verbose=True,
    n_jobs=1,
)
