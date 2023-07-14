from typing import List
import pandas as pd
import numpy as np

from fairlearn.reductions import ExponentiatedGradient

class CustomExponentiatedGradient(ExponentiatedGradient):
    """Custom class to allow for scikit-learn-compatible interface.

    Specifically, this method takes (and ignores) a sample_weights
    parameter to its .fit() method; otherwise identical to
    fairlearn.ExponentiatedGradient.
    """

    def __init__(self, sensitive_features: List[str], **kwargs):
        super().__init__(**kwargs)
        self.sensitive_features = sensitive_features

    def fit(self, X, y, **kwargs):
        if isinstance(X, pd.DataFrame):
            super().fit(X.values, y.values,
                        sensitive_features=X[self.sensitive_features].values,
                        **kwargs)
        elif isinstance(X, np.ndarray):
            super().fit(X, y, sensitive_features = X[:, self.sensitive_features], **kwargs)
        else:
            raise NotImplementedError

    def predict_proba(self, X):
        """Alias to _pmf_predict(). Note that this tends to return 'hard'
        predictions, which don't perform well for metrics like cross-entropy."""
        return super()._pmf_predict(X)
