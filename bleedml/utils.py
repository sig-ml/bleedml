from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np

def scan1D(X, y, window=100, estimator_params=dict(n_jobs=-1), cv=3):
    "Sliding scanner for variable length input samples"
    inputs, labels, instances = [], [], []
    instance_count = 0
    for sample, label in zip(X, y):
        sample_len = len(sample)
        for s in range(sample_len-window):
            inputs.append(sample[s: s+window])
            labels.append(label)
            instances.append(instance_count)
        instance_count += 1
    rf = RandomForestClassifier(**estimator_params)
    estimator_params.update({'max_features': 1})
    cf = RandomForestClassifier(**estimator_params)
    probas1 = cross_val_predict(rf, inputs, labels, cv=cv, method='predict_proba')
    probas2 = cross_val_predict(cf, inputs, labels, cv=cv, method='predict_proba')
    probas = []
    for instance in set(instances):
        mask = [i == instance for i in instances]
        p1 = probas1[mask]
        p2 = probas2[mask]
        p = np.concatenate([p1.flatten(), p2.flatten())], axis=0)
        probas.append(p)
    return probas

def scan2D(X, y, window=(10, 10), estimator_params=dict(n_jobs=-1), cv=3):
    "2D scanning"
    inputs, labels, instances = [], [], []
    instance_count = 0
    for sample, label in zip(X, y):
        sample_shape = sample.shape
        for s1 in range(sample.shape[0]-window[0]):
            for s2 in range(sample.shape[1]-window[1]):
                part = sample[s1:s1+window[0], s2:s2+window[1]]
                inputs.append(part)
                labels.append(label)
    rf = RandomForestClassifier(**estimator_params)
    estimator_params.update({'max_features': 1})
    cf = RandomForestClassifier(**estimator_params)
    probas1 = cross_val_predict(rf, inputs, labels, cv=cv, method='predict_proba')
    probas2 = cross_val_predict(cf, inputs, labels, cv=cv, method='predict_proba')
    probas = []
    for instance in set(instances):
        mask = [i == instance for i in instances]
        p1 = probas1[mask]
        p2 = probas2[mask]
        p = np.concatenate([p1.flatten(), p2.flatten())], axis=0)
        probas.append(p)
    return probas

def multi1Dscan(X, y, windows=[10, 50, 100], estimator_params=dict(n_jobs=-1), cv=3):
    "Multi grained scanner"
    scans = [scan1D(X, y, window, estimator_params, cv=cv) for window in windows]
    inputs = np.concatenate(scans, axis=1)
    return inputs

def multi2Dscan(X, y, windows=[(10, 10), (50, 50), (100, 100)],
        estimator_params=dict(n_jobs=-1), cv=3):
    "Mltiple window 2D scan"
    scans = [scan2D(X, y, window, estimator_params, cv) for window in windows]
    inputs = np.concatenate(scans, axis=1)
    return inputs
