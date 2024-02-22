"""Train german model."""
import sys
from os.path import dirname, abspath

import numpy as np
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)

from data.processing_functions import get_and_preprocess_german_short

german_data, categorical_mapping = get_and_preprocess_german_short()


def make_categorical_col_ids(data, col_name):
    """
    Make a list of column ids for the categorical columns.
    """
    return [list(np.sort(data['x_values'][col_name].unique()))]


# savings_categories = [['NA', 'little', 'moderate', 'quite rich', 'rich']]
age_categories = [[0, 1, 2, 3]]
savings_categories = [[0, 1, 2, 3, 4]]
# checking_categories = [['NA', 'little', 'moderate', 'rich']]
checking_categories = [[0, 1, 2, 3]]
job_level_categories = [[0, 1, 2, 3]]
one_hot_col_names = ['Gender', 'Housing Type', 'Credit Purpose']
sex_categories = make_categorical_col_ids(german_data, 'Gender')
housing_categories = make_categorical_col_ids(german_data, 'Housing Type')
purpose_categories = make_categorical_col_ids(german_data, 'Credit Purpose')

standard_scaler_col_list = ['Credit amount', 'Credit Duration']

X_values = german_data["x_values"]
y_values = german_data["y_values"]

# Transform categorical names to according int values
scalar = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X_values, y_values, test_size=0.20)
# Save data before transformations
X_train['y'] = y_train
X_test['y'] = y_test
X_train.to_csv('german_train.csv')
X_test.to_csv('german_test.csv')
X_train.pop("y")
X_test.pop("y")

X_train = X_train.values
X_test = X_test.values

# Setup pipeline
# lr_pipeline = Pipeline([('scaler', StandardScaler()),
#                         ('lr', LogisticRegression(C=1.0, max_iter=10_000))])
# save_column_id_to_possible_values_mapping(X_train, categorical_col_ids)
# save_column_id_to_value_index_mapping(X_train, categorical_col_ids)


data_columns = german_data['column_names']
preprocessor = ColumnTransformer(
    [
        ('onehot_sex', OneHotEncoder(categories=sex_categories), [data_columns.index('Gender')]),
        ('onehot_housing', OneHotEncoder(categories=housing_categories), [data_columns.index('Housing Type')]),
        ('onehot_purpose', OneHotEncoder(categories=purpose_categories), [data_columns.index('Credit Purpose')]),
        ('ordinal_saving', OrdinalEncoder(categories=savings_categories), [data_columns.index('Saving accounts')]),
        ('ordinal_checking', OrdinalEncoder(categories=checking_categories), [data_columns.index('Checking account')]),
        ('ordinal_age', OrdinalEncoder(categories=age_categories), [data_columns.index('Age Group')]),
        ('ordinal_job', OrdinalEncoder(categories=job_level_categories), [data_columns.index('Job Level')]),
        ('scaler', StandardScaler(), [data_columns.index(col) for col in standard_scaler_col_list])
    ],
    remainder='drop'
)
lr_pipeline = Pipeline([('preprocessing', preprocessor),
                        ('lr', GradientBoostingClassifier())])
lr_pipeline.fit(X_train, y_train)

print("Train Score:", lr_pipeline.score(X_train, y_train))
print("Test Score:", lr_pipeline.score(X_test, y_test))
print("Portion y==1:", np.sum(y_test == 1)
      * 1. / y_test.shape[0])

x_test_pred = lr_pipeline.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, x_test_pred, pos_label=1)
print("AUC_test", metrics.auc(fpr, tpr))

with open("german_model_short_grad_tree.pkl", "wb") as f:
    pkl.dump(lr_pipeline, f)

print("Saved model!")
