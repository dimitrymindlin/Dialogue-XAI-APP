# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data.ml_utilities import label_encode_and_save_classes, construct_pipeline, train_model
from data.train_utils import (
    setup_training_environment, save_train_test_data, setup_feature_display_names,
    apply_display_names, save_config_file, save_categorical_mapping,
    save_model_and_features, evaluate_model, update_config_with_metrics,
    print_feature_importances, get_and_store_feature_importances, generate_feature_names_output, check_nan_values
)

DATASET_NAME = "adult"
config_path = f"./{DATASET_NAME}_model_config.json"
save_path = f"./{DATASET_NAME}"
save_flag = True

# Mapping of categories to new integer representations
category_to_int = {
    'Primary Education': 1,
    'Middle School': 2,
    'High School': 3,
    'Undergraduate Level': 4,
    'Masters Level': 5,
    'Doctorate/Prof Level': 6
}


def map_capital_col(data, config):
    """Refine Investment Outcome with categories including Break-Even and No Activity."""
    # Initialize default category for all as 'No Activity (0$)'
    data['InvestmentOutcome'] = 'No Investment'

    # Calculate Investment Outcome for other cases
    investment_outcome = data['Capital.gain'] - data['Capital.loss']

    # Define bins and labels excluding the default 'No Activity' and 'Break-Even'
    bins = [-float('inf'), -1000, 0, 5000, float('inf')]
    labels = ['Major Loss (more than 1k$)', 'Minor Loss (up to 1k$)', 'Minor Gain (up to 5k$)',
              'Major Gain (above 5k$)']
    # Update categories based on bins and labels for non-default cases
    non_default_cases = investment_outcome != 0
    categorized_outcomes = pd.cut(investment_outcome[non_default_cases], bins=bins, labels=labels)
    data.loc[non_default_cases, 'InvestmentOutcome'] = categorized_outcomes

    # Updated mapping with new categories
    mapping = {
        "Major Loss (more than 1k$)": 0,
        "Minor Loss (up to 1k$)": 1,
        "No Investment": 2,
        "Minor Gain (up to 5k$)": 3,
        "Major Gain (above 5k$)": 4
    }

    # Add the new mapping to the config
    config["ordinal_mapping"]["InvestmentOutcome"] = mapping

    # Optionally, you may drop the original capital gain and loss columns
    data.drop(columns=['Capital.gain', 'Capital.loss'], inplace=True)


def map_education_to_int(education_str):
    """
    Maps an education string to its corresponding broad education category integer.

    Parameters:
    - education_str: A string representing the education level.

    Returns:
    - An integer representing the broad education category.
    """
    # Mapping of education levels to categories
    category_map = {
        'Preschool': 'Dropout',
        '1st-4th': 'Dropout',
        '5th-6th': 'Dropout',
        '7th-8th': 'Dropout',
        '9th': 'Dropout',
        '10th': 'Dropout',
        '11th': 'Dropout',
        '12th': 'Dropout',
        'HS-grad': 'High School grad',
        'Some-college': 'High School grad',
        'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates',
        'Bachelors': 'Bachelors grad',
        'Masters': 'Masters grad',
        'Prof-school': 'Professional Degree',
        'Doctorate': 'Doctorate/Prof Level'
    }

    # Get the category from the education level
    category = category_map.get(education_str, "Unknown")
    return category


def map_education_levels(data, config):
    # Apply binning to education.num
    def refined_bin_education_level(x):
        if x in [1, 2, 3, 4, 5, 6, 7, 8]:
            return 0  # Dropout
        elif x in [9, 10]:
            return 1  # High School grad
        elif x in [11, 12]:
            return 2  # Associates
        elif x == 13:
            return 3  # Bachelors
        elif x == 14:
            return 4  # Masters
        elif x == 15:
            return 5  # Professional Degree
        elif x == 16:
            return 6  # Doctorate/Prof Level
        else:
            return None  # For any value out of the original range

    data['EducationLevel'] = data['Education.num'].apply(refined_bin_education_level)
    data.drop(['Education.num'], axis=1, inplace=True)

    refined_binned_category_map = {
        0: 'Primary Education',
        1: 'Middle School',
        2: 'High School without Graduation',
        3: 'High School Graduate',
        4: 'College without Degree',
        5: "Associate's Degrees",
        6: "Bachelor's Degree",
        7: 'Post-graduate Education'
    }

    # Reverse the mapping for the config
    refined_binned_category_map = {v: k for k, v in refined_binned_category_map.items()}

    # Add the mapping to the config
    config["ordinal_mapping"]["EducationLevel"] = refined_binned_category_map


def standardize_column_names(data):
    data.columns = [col.capitalize() for col in data.columns]


def add_work_life_balance(data):
    categories = ['Poor', 'Fair', 'Good']
    mean = categories.index('Fair')  # Mean set to index of 'Fair' for Gaussian center
    std_dev = 0.75  # Standard deviation, adjust as needed for spread

    # Generate indices from a Gaussian distribution
    gaussian_indices = np.random.normal(loc=mean, scale=std_dev, size=len(data))

    # Clip the indices to lie within the range of categories to avoid out-of-bounds indices
    gaussian_indices_clipped = np.clip(gaussian_indices, 0, len(categories) - 1)

    # Round indices to nearest integer to use as valid category indices
    category_indices = np.round(gaussian_indices_clipped).astype(int)

    # Assign categories based on Gaussian-distributed indices
    data['WorkLifeBalance'] = [categories[i] for i in category_indices]

    # Create a mapping from string values to integers
    mapping = {category: i for i, category in enumerate(categories)}
    # Apply the mapping to the 'WorkLifeBalance' column
    data['WorkLifeBalance'] = data['WorkLifeBalance'].map(mapping)
    return mapping


def fil_nans_with_mode(data):
    data[data == '?'] = np.nan  # Replace '?' with NaN
    cols_with_nan_names_list = data.columns[data.isna().any()].tolist()
    for col in cols_with_nan_names_list:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Check outliars in work hours (working with std dev)
    mean = data['Hours.per.week'].mean()
    std = data['Hours.per.week'].std()
    data_cleaned = data[(data['Hours.per.week'] > mean - 2 * std) & (data['Hours.per.week'] < mean + 2 * std)]
    print(f"Removed {len(data) - len(data_cleaned)} outliars from the data")
    return data_cleaned


def map_ordinal_columns(data, config):
    for col, mapping in config["ordinal_mapping"].items():
        if col in ["WorkLifeBalance", "EducationLevel"]:
            continue
        data[col] = data[col].map(mapping)


def bin_age_column(data):
    bins = [17, 25, 50, 100]
    labels = ["Young (18-25)", "Adult (25-50)", "Old (50-100)"]
    data["Age"] = pd.cut(data["Age"], bins=bins, labels=labels)


def bin_workclass_col(data):
    # Define the mapping from original labels to new categories
    workclass_map = {
        'Private': 'Private Sector',
        'State-gov': 'Government Employment',
        'Federal-gov': 'Government Employment',
        'Self-emp-not-inc': 'Self-Employed / Entrepreneurial',
        'Self-emp-inc': 'Private Sector',
        'Local-gov': 'Government Employment',
        'Without-pay': 'Unemployed / Other',
        'Never-worked': 'Unemployed / Other'
    }

    # Apply the mapping to the "Workclass" column
    data["Workclass"] = data["Workclass"].map(workclass_map)
    return data


def map_race_col(data):
    # print counts of unique race values
    print(data["Race"].value_counts())
    # Only take "white" as race and remove the rest
    # data = data[data["Race"] == "White"]
    # Drop the "Race" column
    data.drop(columns=["Race"], inplace=True)
    return data


def map_country_col(data):
    countries = np.array(data['Native.country'].unique())
    countries = np.delete(countries, 0)
    data['Native.country'].replace(countries, 'Other', inplace=True)
    # Delete other countries records
    # data = data[data['Native.country'] == 'United-States']
    data.drop(columns=['Native.country'], inplace=True)
    return data


def map_occupation_col(data):
    # Mapping of professions to their respective subgroups
    occupation_map = {
        "Adm-clerical": "Admin",
        "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar",
        "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar",
        "Handlers-cleaners": "Blue-Collar",
        "Machine-op-inspct": "Blue-Collar",
        "Other-service": "Service",
        "Priv-house-serv": "Service",
        "Prof-specialty": "Professional",
        "Protective-serv": "Service",
        "Sales": "Sales",
        "Tech-support": "Service",
        "Transport-moving": "Blue-Collar",
    }

    # Apply mapping to the occupation column
    data['Occupation'] = data['Occupation'].map(occupation_map)


def preprocess_marital_status(data):
    data["Marital.status"] = data["Marital.status"].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'],
                                                            'Single')
    data["Marital.status"] = data["Marital.status"].replace(
        ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')


def preprocess_data_specific(data, config):
    if "drop_columns" in config:
        data.drop(columns=config["drop_columns"], inplace=True)

    from data.train_utils import standardize_column_names
    data = standardize_column_names(data)

    # Following functions assume capitalized col names
    config["ordinal_mapping"]['WorkLifeBalance'] = add_work_life_balance(data)
    preprocess_marital_status(data)
    map_education_levels(data, config)
    map_capital_col(data, config)
    map_occupation_col(data)
    map_ordinal_columns(data, config)

    data = fil_nans_with_mode(data)

    # Check which column has "Other" values
    for col in data.columns:
        if "Other" in data[col].unique():
            print(f"Column '{col}' has 'Other' values.")

    target_col = config["target_col"]
    X = data.drop(columns=[target_col])
    y = data[target_col]

    if "columns_to_encode" in config:
        X, encoded_classes = label_encode_and_save_classes(X, config)
    else:
        encoded_classes = {}

    return X, y, encoded_classes


def main():
    # Set up environment and load config
    config, save_path = setup_training_environment(DATASET_NAME, config_path)

    data = pd.read_csv(config["dataset_path"])

    # Store original column names before any preprocessing
    original_columns = list(data.columns)

    # Perform preprocessing without applying rename_columns
    saved_rename_columns = None
    if "rename_columns" in config:
        # Temporarily save and remove rename_columns to prevent premature renaming
        saved_rename_columns = config["rename_columns"]
        config["rename_columns"] = {}

    X, y, encoded_classes = preprocess_data_specific(data, config)

    # Restore rename_columns
    if saved_rename_columns is not None:
        config["rename_columns"] = saved_rename_columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    target_col = config["target_col"]
    # Add labels to X and save train and test data
    X_train[target_col] = y_train
    X_test[target_col] = y_test

    # Save training and test data
    save_train_test_data(X_train, X_test, DATASET_NAME, save_path)

    # Set up feature display names and update config
    feature_display_names, config = setup_feature_display_names(X_train, config, target_col)

    # Apply display names to the dataframes
    X_train, X_test = apply_display_names(X_train, X_test, feature_display_names)

    # Save the updated config
    save_config_file(config, save_path, DATASET_NAME)

    # Create a copy of dataframes for later reference
    X_train_with_names = X_train.copy()
    X_test_with_names = X_test.copy()

    # Save categorical mapping from preprocess function
    categorical_mapping = {}
    for feature in config["columns_to_encode"]:
        # Create mappings based on encoded_classes
        if feature in encoded_classes:
            categorical_mapping[feature] = {str(v): k for k, v in encoded_classes[feature].items()}

    # Save categorical mapping
    save_categorical_mapping(categorical_mapping, save_path)

    # Remove target column for model training
    X_train.drop(columns=[target_col], inplace=True)
    X_test.drop(columns=[target_col], inplace=True)

    # Check for NaN values
    check_nan_values(X_train)

    # Change list of column names to be encoded to a list of column indices
    # Map column names to their display names for index lookup
    columns_for_encoding = []
    for feature in config["columns_to_encode"]:
        display_name = feature_display_names.get_display_name(feature)
        if display_name in X_train.columns:
            columns_for_encoding.append(display_name)
        elif feature in X_train.columns:
            columns_for_encoding.append(feature)
        else:
            print(f"Warning: Column '{feature}' not found in DataFrame")
    
    columns_to_encode = [X_train.columns.get_loc(col) for col in columns_for_encoding]
    pipeline = construct_pipeline(columns_to_encode, RandomForestClassifier())

    # Set up model parameters
    model_params = {("model__" + key if not key.startswith("model__") else key): value
                    for key, value in config["model_params"].items()}
    search_params = config.get("random_search_params", {"n_iter": 10, "cv": 5, "random_state": 42})

    # Train model
    best_model, best_params = train_model(X_train, y_train, pipeline, model_params, search_params)
    best_model.fit(X_train, y_train)

    # Evaluate model
    train_score, test_score = evaluate_model(best_model, X_train, X_test, y_train, y_test)
    print("Best Parameters:", best_params)

    # Update config with metrics
    config = update_config_with_metrics(config, train_score, test_score)

    # Save updated config with metrics
    save_config_file(config, save_path, DATASET_NAME)

    # Save model and feature information
    if save_flag:
        save_model_and_features(best_model, X_train, config, save_path, DATASET_NAME)

    # Generate feature names output after transformation
    feature_names_out = generate_feature_names_output(X_train, config, categorical_mapping)

    # Get and store feature importances in config
    feature_importances = get_and_store_feature_importances(best_model, feature_names_out, config, save_path, DATASET_NAME)

    print("Saved model!")


if __name__ == "__main__":
    main()
