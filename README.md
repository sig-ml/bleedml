BleedML
=======

The aim is o have a pip installable library which implements as may of the "cool" / "latest" / "bleeding edge" algorithms in ML as possible.


Installation
------------

`pip install bleedml`

Usage
-----

Most of the algorithms are meant as a drop in replacement for `sklearn` estimators.

```python
from bleedml.classifiers import CascadeForest
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True)
est = CascadeForest()
scores = cross_val_score(est, X, y, cv=3, scoring='accuracy')
print(scores.mean())
```


Algorithms Available
--------------------

- Classifiers
    - [x] **CascadeForests** [Deep Learning Forests](https://arxiv.org/abs/1702.08835)
    - [ ] **MCCVM** [Multi Class Core Vector Machine](https://dl.acm.org/citation.cfm?id=1273502)
- Utils
    - [x] **scan1D** [1D windowed scanning] (https://arxiv.org/abs/1702.08835)
    - [x] **scan2D** [2D windowed scanning] (https://arxiv.org/abs/1702.08835)
    - [x] **multi1Dscan** [Multigrain 1D windowed scanning] (https://arxiv.org/abs/1702.08835)
    - [x] **multi2Dscan** [Multigrain 2D windowed scanning] (https://arxiv.org/abs/1702.08835)
    - [x] **get_meb** [Minimum Enclosing Ball](https://dl.acm.org/citation.cfm?id=644240)
