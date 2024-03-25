import numpy as np
from pymoo.termination import get_termination
import matplotlib.pyplot as plt

def train(est, X_train, X_prime_train, y_train, X_test, sens_cols,**kwargs):

    protected_features = [c for c in X_train.columns if any(s in c for s in sens_cols)]
    print('protected_features:',protected_features)
    
    est.fit(
        X_train, y_train, 
        protected_features=protected_features,
        abs_val=True,
        termination=('n_gen',50)
    )
    train_predictions = est.predict_archive(X_train)
    train_probabilities = est.predict_proba_archive(X_train)
    test_predictions = est.predict_archive(X_test)
    test_probabilities = est.predict_proba_archive(X_test)
    history = est.res_.history
    best_est = history[-1].opt[est.I_]

    return (
        train_predictions,
        test_predictions,
        train_probabilities,
        test_probabilities,
        history,
        best_est
    )
