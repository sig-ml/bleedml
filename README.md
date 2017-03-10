BleedML
=======

`Aim`: To have a pip installable library which implements as may of the "cool" / "latest" / "bleeding edge" algorithms in ML as possible.


Installation
------------

`pip install bleedml`

Usage
-----

Most of the algorithms are meant as a drop in replacement for `sklearn` estimators.

```python
from bleedml.classifiers import GCForest
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True)
est = GCForest()
scores = cross_val_score(est, X, y, cv=3, scoring='accuracy')
print(scores.mean())
```


Algorithms Available
--------------------

- [x] **gcForests** [Deep Learning Forests](https://arxiv.org/abs/1702.08835)
