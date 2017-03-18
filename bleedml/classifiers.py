from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_predict, cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class CascadeForest(BaseEstimator, ClassifierMixin):
    "From https://arxiv.org/abs/1702.08835"
    def __init__(self, cv=3, n_estimators=100, scoring=accuracy_score, tolerance=0.01, validation_fraction=0.8):
        """
        cv                  :(3) how many folds to use for prediction generation in each layer?
        n_estimators        :(100) how many estimators in each layer's forest
        scoring             :(accuracy) what score to minimize for?
        tolerance           :(0.01) what tolerance to stop at?
        validation_fraction :(0.8) what fraction of data to use for validation during cascade growth?
        """
        self.n_estimators = n_estimators
        self.cv = cv
        self.scoring = scoring
        self.tolerance = tolerance
        self.validation_fraction = validation_fraction

    def fit(self, X, y):
        # Check data
        X, y = check_X_y(X, y)
        # Split to grow cascade and validate
        mask = np.random.random(y.shape[0]) < self.validation_fraction
        X_tr, X_vl = X[mask], X[~mask]
        y_tr, y_vl = y[mask], y[~mask]

        self.classes_ = unique_labels(y)
        self.layers_, inp_tr, inp_vl = [], X_tr, X_vl
        self.scores_ = []

        # First layer
        forests = [RandomForestClassifier(max_features=1, n_estimators=self.n_estimators, min_samples_split=10, criterion='gini', n_jobs=-1),  # Complete random
                    RandomForestClassifier(max_features=1, n_estimators=self.n_estimators, min_samples_split=10, criterion='gini', n_jobs=-1),  # Complete random
                    RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1),
                    RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1)]
        _ = [f.fit(inp_tr, y_tr) for f in forests]
        p_vl = [f.predict_proba(inp_vl) for f in forests]
        labels = [self.classes_[i] for i in np.argmax(np.array(p_vl).mean(axis=0), axis=1)]
        score = self.scoring(y_vl, labels)
        self.layers_.append(forests)
        self.scores_.append(score)
        p_tr = [cross_val_predict(f, inp_tr, y_tr, cv=self.cv, method='predict_proba') for f in forests]

        # Fit other layers
        last_score = score
        inp_tr, inp_vl = np.concatenate([X_tr]+p_tr, axis=1), np.concatenate([X_vl]+p_vl, axis=1)
        while True:  # Grow cascade
            forests = [RandomForestClassifier(max_features=1, n_estimators=self.n_estimators, min_samples_split=10, criterion='gini', n_jobs=-1),  # Complete random
                    RandomForestClassifier(max_features=1, n_estimators=self.n_estimators, min_samples_split=10, criterion='gini', n_jobs=-1),  # Complete random
                    RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1),
                    RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1)]
            _ = [forest.fit(inp_tr, y_tr) for forest in forests] # Fit the forest
            p_vl = [forest.predict_proba(inp_vl) for forest in forests]
            labels = [self.classes_[i] for i in np.argmax(np.array(p_vl).mean(axis=0), axis=1)]
            score = self.scoring(y_vl, labels)

            if score - last_score > self.tolerance:
                self.layers_.append(forests)
                p_tr = [cross_val_predict(f, inp_tr, y_tr, cv=self.cv, method='predict_proba') for f in forests]
                inp_tr, inp_vl = np.concatenate([X_tr]+p_tr, axis=1), np.concatenate([X_vl]+p_vl, axis=1)
                self.scores_.append(score)
                last_score = score
                print(self.scores_)
            else:
                break
        # Retrain on entire dataset
        inp_ = X
        for forests in self.layers_:
            _ = [f.fit(inp_, y) for f in forests]
            p = [cross_val_predict(f, inp_, y, cv=self.cv, method='predict_proba') for f in forests]
            inp_ = np.concatenate([X]+p, axis=1)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['layers_', 'classes_', 'scores_'])
        X = check_array(X)
        inp_ = X
        for forests in self.layers_:
            p = [f.predict_proba(inp_) for f in forests]
            inp_ = np.concatenate([X] + p, axis=1)
        avg_p = np.array(p).mean(axis=0)
        return avg_p

    def predict(self, X):
        X = check_array(X)
        avg_p = self.predict_proba(X)
        labels = np.array([self.classes_[i] for i in np.argmax(avg_p, axis=1)])
        return labels

if __name__ == '__main__':
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(CascadeForest)
