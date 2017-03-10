from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_predict, cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class GCForest(BaseEstimator, ClassifierMixin):
    "From https://arxiv.org/abs/1702.08835"
    def __init__(self, n_layers=3, cv=3, n_estimators_in_layers=[[1000]*4, [1000]*4, [1000]*4]):
        self.n_estimators_in_layers = n_estimators_in_layers
        self.n_layers = n_layers
        self.cv = cv

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        layers, input_data = [], X
        classes = unique_labels(y)
        for i in range(self.n_layers):
            f1est, f2est, f3est, f4est = self.n_estimators_in_layers[i]
            forests = [RandomForestClassifier(max_features=1, n_estimators=f1est, min_samples_split=10),
                    RandomForestClassifier(max_features=1, n_estimators=f2est, min_samples_split=10),
                    RandomForestClassifier(n_estimators=f3est),
                    RandomForestClassifier(n_estimators=f4est)]
            predictions = [cross_val_predict(f, input_data, y, method='predict_proba', cv=self.cv) for f in forests]
            _ = [forest.fit(input_data, y) for forest in forests] # Fit the forest
            layers.append(forests)
            input_data = np.concatenate([X] + predictions, axis=1)
        self.layers_ = layers

    def predict_proba(self, X):
        check_is_fitted(self, ['layers_'])
        X = check_array(X)
        input_data = X
        for forests in self.layers_:
            predictions = [forest.predict_proba(input_data) for forest in forests]
            input_data = np.concatenate([X] + predictions, axis=1)
        average_predictions = np.array(predictions).mean(axis=0)
        return average_predictions

    def predict(self, X):
        check_is_fitted(self, ['layers_'])
        average_predictions = self.predict_proba(X)
        labels_indices = np.argmax(average_predictions, axis=1)
        labels = [self.classes_[i] for i in labels_indices]
        return labels
