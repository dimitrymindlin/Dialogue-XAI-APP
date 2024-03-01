# ml_utilities.py
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
import os


def label_encode_and_save_classes(df, columns, save_path):
    encoded_classes = {}
    categorical_mapping = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        # Create a dictionary mapping for each column
        mapping_dict = {int(i): str(class_name) for i, class_name in enumerate(le.classes_)}
        encoded_classes[col] = mapping_dict
        # categorical_mapping: map from integer to list of strings, names for each
        # value of the categorical features.
        col_id = df.columns.get_loc(col)
        categorical_mapping[col_id] = list(mapping_dict.values())

    # Save all the mappings in separate JSON files
    with open(os.path.join(save_path, "encoded_col_mapping.json"), 'w') as f:
        json.dump(encoded_classes, f)
    with open(os.path.join(save_path, "categorical_mapping.json"), 'w') as f:
        json.dump(categorical_mapping, f)

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
