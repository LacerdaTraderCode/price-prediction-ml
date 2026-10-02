"""The two learned models under comparison: a tree ensemble and a small neural network.

The neural network here is scikit-learn's `MLPClassifier` rather than a
PyTorch or Keras model: for a handful of engineered tabular features and a
few thousand rows, a small multi-layer perceptron is the right amount of
model for the problem, and keeping the whole pipeline on scikit-learn means
it runs anywhere with no heavy framework or GPU expectation.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0)


def build_neural_network() -> Pipeline:
    # MLPClassifier is sensitive to feature scale, unlike the tree ensemble
    # above, so it is always paired with a scaler rather than used alone.
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", max_iter=2000, random_state=0),
    )
