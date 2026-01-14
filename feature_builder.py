# feature_builder.py
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureBuilderNoDelayRate(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()
        self.length_median_ = None
        if "Length" in X.columns:
            self.length_median_ = float(pd.to_numeric(X["Length"], errors="coerce").median())
        return self

    def transform(self, X):
        X = X.copy()

        if "Route" not in X.columns and ("AirportFrom" in X.columns and "AirportTo" in X.columns):
            X["Route"] = X["AirportFrom"].astype(str) + "-" + X["AirportTo"].astype(str)

        if "AirlineRoute" not in X.columns:
            if "Airline" in X.columns and "Route" in X.columns:
                X["AirlineRoute"] = X["Airline"].astype(str) + "_" + X["Route"].astype(str)

        if "DayOfWeek" in X.columns and "IsWeekend" not in X.columns:
            X["IsWeekend"] = X["DayOfWeek"].isin([6, 7]).astype(int)

        if "Length" in X.columns and self.length_median_ is not None and "IsLongFlight" not in X.columns:
            L = pd.to_numeric(X["Length"], errors="coerce")
            X["IsLongFlight"] = (L > self.length_median_).astype(int)

        return X
