"""k-NN baseline model: shared preprocessing followed by a k-nearest-neighbours classifier."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from cmn.data import build_preprocessor


def build_pipeline(features, k):
    """z-score numerics + one-hot categoricals, then k-NN (Euclidean distance, majority vote)."""
    return Pipeline([("prep", build_preprocessor(features)),
                     ("knn", KNeighborsClassifier(n_neighbors=k))])
