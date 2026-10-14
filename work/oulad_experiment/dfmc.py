"""Minimal, audited DFMC calibrator for the OULAD pilot.

Based on the algorithm in ``src/alg/Boosting.py`` of
https://github.com/hjenryin/Discretization-free-MC (NeurIPS 2025).
The original MIT license is reproduced at the end of this file.

Only NumPy and LightGBM are required.  No source-package setup, notebook,
PyTorch import, or hidden train/validation split is performed here.

``predict`` intentionally returns RAW p + tree correction, including values
outside [0, 1].  Clip copies of those values when computing probability metrics;
retain the raw values to report the frequency of out-of-range predictions.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb


class DFMC:
    """Depth-two tree-ensemble residual calibration with group indicators.

    Parameters can be supplied in a mapping and/or as keyword arguments.
    Accepted convenient names include ``n_estimators``, ``num_leaves``,
    ``max_depth``, ``min_child_samples``, ``learning_rate``, ``reg_lambda``,
    ``n_jobs``, and ``seed``; native LightGBM names are also accepted.

    Use explicit historical validation arrays for early stopping.  Without
    them all supplied calibration rows are used, for a fixed number of trees.
    The caller must construct independent calibration/validation sets (for
    repeated observations, split by student rather than by row).
    """

    _DEFAULTS = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "max_depth": 2,
        # Four leaves is the maximum permitted by the original depth-two tree.
        "num_leaves": 4,
        "min_data_in_leaf": 20,
        "lambda_l2": 0.0,
        "max_bin": 8192,
        "min_data_in_bin": 1,
        "num_threads": 8,
        "verbosity": -1,
        "seed": 42,
        "num_boost_round": 2000,
        "early_stopping_rounds": 50,
    }

    _ALIASES = {
        "boost": "boosting_type",
        "boosting": "boosting_type",
        "eta": "learning_rate",
        "subsample": "bagging_fraction",
        "subsample_freq": "bagging_freq",
        "n_estimators": "num_boost_round",
        "num_iterations": "num_boost_round",
        "min_child_samples": "min_data_in_leaf",
        "reg_lambda": "lambda_l2",
        "n_jobs": "num_threads",
        "random_state": "seed",
        "verbose": "verbosity",
        "early_stopping_round": "early_stopping_rounds",
    }

    def __init__(self, params=None, **kwargs):
        supplied = {} if params is None else dict(params)
        supplied.update(kwargs)
        normalized = {}
        for name, value in supplied.items():
            canonical = self._ALIASES.get(name, name)
            if canonical in normalized and normalized[canonical] != value:
                raise ValueError(f"Conflicting aliases for {canonical}")
            normalized[canonical] = value
        self.params = dict(self._DEFAULTS)
        self.params.update(normalized)
        if self.params["objective"] not in {"regression", "regression_l2", "l2"}:
            raise ValueError("DFMC requires the squared-error regression objective")
        if self.params["boosting_type"] != "gbdt":
            raise ValueError("This DFMC implementation requires gbdt boosting")
        self.num_boost_round = int(self.params["num_boost_round"])
        self.early_stopping_rounds = int(self.params["early_stopping_rounds"])
        if self.num_boost_round < 1 or self.early_stopping_rounds < 0:
            raise ValueError("Invalid number of boosting/early-stopping rounds")
        self.model = None

    @staticmethod
    def _vector(values, name):
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite one-dimensional array")
        return array

    def _features(self, p, G, *, fitting=False):
        p = self._vector(p, "p")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("p must contain probabilities in [0, 1]")
        groups = np.asarray(G, dtype=np.float64)
        if groups.ndim == 1:
            groups = groups.reshape(-1, 1)
        if groups.ndim != 2 or groups.shape[0] != len(p):
            raise ValueError("G must have one row per probability")
        if not np.all(np.isfinite(groups)) or np.any((groups != 0) & (groups != 1)):
            raise ValueError("G must contain finite binary group indicators")
        columns = tuple(G.columns) if hasattr(G, "columns") else None
        if fitting:
            self.n_groups_ = groups.shape[1]
            self.group_columns_ = columns
        else:
            if groups.shape[1] != self.n_groups_:
                raise ValueError("G has a different number of group columns")
            if self.group_columns_ is not None and columns is not None:
                if columns != self.group_columns_:
                    raise ValueError("G columns differ from the fitting order")
        # Official source adds a clipped probability feature after group columns.
        X = np.column_stack((groups, np.clip(p, 0.0, 1.0)))
        return p, X

    def _labels(self, y, n):
        y = self._vector(y, "y")
        if len(y) != n or np.any((y < 0) | (y > 1)):
            raise ValueError("y must have one label in [0, 1] per probability")
        return y

    def fit(self, p, y, G, p_val=None, y_val=None, G_val=None):
        """Fit corrections on calibration data, optionally with validation.

        ``label=y, init_score=p`` is the official implementation's way to
        fit residuals under squared loss.  The base scores are supplied again
        at prediction time, because LightGBM does not save row-wise init_score
        in the trained predictor.
        """
        if self.model is not None:
            raise ValueError("Already fitted; create a new DFMC instance")
        p, X = self._features(p, G, fitting=True)
        if not len(p):
            raise ValueError("Calibration set must not be empty")
        y = self._labels(y, len(p))
        supplied_validation = (p_val is not None, y_val is not None, G_val is not None)
        if any(supplied_validation) and not all(supplied_validation):
            raise ValueError("Supply p_val, y_val and G_val together")
        train = lgb.Dataset(X, label=y, init_score=p, free_raw_data=False)
        valid_sets, valid_names = [train], ["train"]
        if all(supplied_validation):
            p_val, X_val = self._features(p_val, G_val)
            if not len(p_val):
                raise ValueError("Validation set must not be empty")
            y_val = self._labels(y_val, len(p_val))
            validation = lgb.Dataset(
                X_val, label=y_val, init_score=p_val,
                reference=train, free_raw_data=False,
            )
            valid_sets.append(validation)
            valid_names.append("validation")
        training_params = dict(self.params)
        training_params.pop("num_boost_round")
        training_params.pop("early_stopping_rounds")
        # Early stopping is explicit and only active with a true validation set.
        first_metric_only = training_params.pop("first_metric_only", True)
        if "skip_validation" in training_params:
            raise ValueError("Use omitted validation arrays instead of skip_validation")
        callbacks = [lgb.log_evaluation(period=0)]
        self.evals_result_ = {}
        callbacks.append(lgb.record_evaluation(self.evals_result_))
        if len(valid_sets) == 2 and self.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(
                self.early_stopping_rounds,
                first_metric_only=first_metric_only, verbose=False,
            ))
        self.model = lgb.train(
            training_params, train, num_boost_round=self.num_boost_round,
            valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
        )
        self.best_iteration_ = self.model.best_iteration or self.model.current_iteration()
        self.n_calibration_ = len(p)
        self.n_validation_ = 0 if p_val is None else len(p_val)
        self.training_params_ = training_params
        return self

    def predict(self, p, G):
        """Return unbounded probabilities (p plus fitted tree correction)."""
        if self.model is None:
            raise ValueError("DFMC has not been fitted")
        p, X = self._features(p, G)
        if len(p) == 0:
            return np.empty(0, dtype=np.float64)
        correction = self.model.predict(X, num_iteration=self.best_iteration_)
        return p + correction


# MIT License
# Copyright (c) 2025 Hongyi Jin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
