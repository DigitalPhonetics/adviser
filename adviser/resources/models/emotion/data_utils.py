import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn import preprocessing
import xarray
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# =========================================================================
#
#  This file is used to load data from netCDF (.nc) files.
#  Written by Fang Bao (2019)
#  Adapted by Michael Neumann, last update: April 2019
#
#  - Arguments: path to the .nc file
#  - Output: feature and label data
#
# =========================================================================


class InputData(object):

    def __init__(self, path):
        self.data = xarray.open_dataset(path)
        self.file_name = self.data.file_name.to_pandas()
        self.feature_name = self.data.feature_name.to_pandas()
        self.feature_value = self.data.feature_value.to_pandas()
        if "category" in self.data.variables:
            self.category = self.data.category.to_pandas()
        if "arousal" in self.data.variables:
            self.arousal = self.data.arousal.to_pandas()
        if "valence" in self.data.variables:
            self.valence = self.data.valence.to_pandas()

    def get_data(self):
        """Get an object of xarray.Dataset"""

        return self.data

    def get_session(self, session_prefix):
        """Get the names of all files which belong to the given session

        @param session_prefix: Prefix that files of one session have
                               in common with, e.g. All the files in
                               session 1 have prefix "Ses01"
        @return session: List of file names in this session
        """
        session = [name for name in self.file_name
                   if name.startswith(session_prefix)]
        print("{} size: {}".format(session_prefix,
                                   np.array(session).shape))
        return session

    def _extract(self, df, indices=None):
        """Extract the items at the given indices from the dataframe.

        If indices not given, all items in the dataframe will be returned
        @param df: Dataframe to be extracted from
        @param indices: Indices of the items to be extracted
        @return df: Dataframe only having the items at the indices
        """
        if indices is not None:
            # return df.loc[df.index.isin(indices)]
            return df.loc[indices]
        return df

    def get_features(self, indices=None):
        """Get feature values of the items at the given indices"""
        return self._extract(self.feature_value, indices)

    def get_category(self, indices=None):
        """Get category of the items at the given indices"""

        return self._extract(self.category, indices)

    def get_arousal(self, indices=None):
        """Get arousal of the items at the given indices"""

        return self._extract(self.arousal, indices)

    def get_valence(self, indices=None):
        """Get valence of the items at the given indices"""

        return self._extract(self.valence, indices)


def normalize_cv(X, y, i, norm="zero_score"):
    X_test = X[i]
    y_test = y[i]
    X_train = pd.concat(X[:i] + X[i+1:])
    y_train = pd.concat(y[:i] + y[i+1:])
    if norm == "min_max":
        scaler = preprocessing.MinMaxScaler()
    elif norm == "max_abs":
        scaler = preprocessing.MaxAbsScaler()
    else:
        scaler = preprocessing.StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                           index=y_train.index.values)
    X_train.columns = X[i].columns.values
    X_test = pd.DataFrame(scaler.transform(X_test), index=y_test.index.values)
    X_test.columns = X[i].columns.values
    return X_train, X_test, y_train, y_test


def normalize(data, norm="zero_score", scaler=None):
    """Normalize pandas Dataframe.

    @param data: Input dataframe
    @param norm: normalization method [default: zero_score standardization],
    alternatives: 'min_max', 'max_abs'
    @return datascaled: normalized dataframe
    """
    if scaler is not None:
        datascaled = pd.DataFrame(scaler.transform(data),
                                  index=data.index.values)
        datascaled.columns = data.columns.values
    else:
        if norm == "min_max":
            scaler = preprocessing.MinMaxScaler()
        elif norm == "max_abs":
            scaler = preprocessing.MaxAbsScaler()
        else:
            scaler = preprocessing.StandardScaler()
        datascaled = pd.DataFrame(scaler.fit_transform(data),
                                  index=data.index.values)
        datascaled.columns = data.columns.values
    return datascaled, scaler


# deprecated - use sklearn.model_selection.train_test_split instead
def split_data(data, splits=[0.7, 0.3], labels=None, abs_splits=None):
    assert sum(splits) == 1.0, "Data splits must sum up to 1.0"
    if not abs_splits:
        total = data.shape[0]
        abs_splits = [int(round(total * i)) for i in splits]

    # recursive function - trivial case: no split, only 1 element
    if len(abs_splits) == 1:
        return [data] if labels is None else [[data], [labels]]
    else:
        # take first split-portion from list and return two DataFrames
        portion = abs_splits.pop(0)
        first = data.sample(n=portion, random_state=42)
        rest = data.drop(first.index)
        if labels is not None:
            f_lbl = labels.reindex(first.index)
            r_lbl = labels.drop(f_lbl.index)
            return(
                [x[0] + x[1] for x in zip(
                    [[first], [f_lbl]],
                    split_data(rest, labels=r_lbl, abs_splits=abs_splits)
                )]
            )
        else:
            return [first] + split_data(rest, abs_splits=abs_splits)
