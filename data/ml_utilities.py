# ml_utilities.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
import os


def label_encode_and_save_classes(df, columns, save_path):
    encoded_classes = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoded_classes[col] = le.classes_
        np.save(os.path.join(save_path, col), le.classes_)
    return df, encoded_classes


def preprocess_data(data, columns_to_encode, ordinal_info, drop_columns=None):
    if drop_columns:
        data.drop(columns=drop_columns, inplace=True)
    if columns_to_encode:
        data, encoded_classes = label_encode_and_save_classes(data, columns_to_encode)
    if ordinal_info:
        for col, mapping in ordinal_info.items():
            data[col] = data[col].map(mapping)
    return data


def construct_pipeline(columns_to_encode, model):
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(), columns_to_encode),
        ], remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('to_array', FunctionTransformer(np.asarray, validate=False)),
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    return pipeline


def train_model(X, y, pipeline, model_params, search_params):
    random_search = RandomizedSearchCV(pipeline, param_distributions=model_params, **search_params)
    random_search.fit(X, y)
    return random_search.best_estimator_, random_search.best_params_
