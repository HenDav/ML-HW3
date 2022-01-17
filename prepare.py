import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler


########################################################
# HELPER FUNCTIONS

def address_to_zipcode(address):
    return address.apply(lambda x: int(x.split(' ')[len(x.split(' ')) - 1]) if pd.notnull(x) else x)


def get_non_nan(column):
    # Get a random sample from column which is not NaN
    x = column.sample().item()
    while type(x) is not str and np.isnan(x):
        x = column.sample().item()
    return x


def get_interval_groups(dataframe, feature, number_of_intervals):
    # Get an array of the points in the feature (ordinal or continuous) according to which we wish to divide our
    # dataset.
    min_in_range = int(dataframe[feature].min(skipna=True))
    max_in_range = int(dataframe[feature].max(skipna=True)) + 1
    delta = int((max_in_range - min_in_range) / number_of_intervals)
    return np.arange(min_in_range, max_in_range + delta, delta)


def impute_from_weight_group(entry, dataframe, feature, number_of_intervals):
    # Impute values in the "feature" column separately for each weight group
    interval_groups = get_interval_groups(dataframe, 'weight', number_of_intervals)
    max_in_entry_group = np.argwhere(interval_groups > entry['weight'])[0][0]
    clean_group = dataframe[(dataframe['weight'] > interval_groups[max_in_entry_group - 1]) &
                            (dataframe['weight'] <= interval_groups[max_in_entry_group])]
    entry[feature] = get_non_nan(clean_group[feature])
    return entry


def impute_feature(dataframe, feature, method):
    # Impute feature in dataframe according to a specific imputation method
    if method == "Median":
        dataframe[feature] = dataframe[feature].fillna(dataframe[feature].agg("median"))
    elif method == "Mean":
        dataframe[feature] = dataframe[feature].fillna(dataframe[feature].agg("mean"))
    elif method == "Binary":
        dataframe[feature] = dataframe[feature].fillna(np.random.randint(0, 1))
    elif method == "by_weight":
        dataframe = dataframe.apply(
            lambda x: impute_from_weight_group(x, dataframe, feature, 10) if pd.isnull(x[feature]) else x, axis=1)
    # else method == "Sample"
    else:
        dataframe[feature] = dataframe[feature].apply(
            lambda x: get_non_nan(dataframe[feature]) if pd.isnull(x) else x)

    return dataframe


def percentile25(dataframe, feature):
    return dataframe[feature].quantile(0.25)


def percentile75(dataframe, feature):
    return dataframe[feature].quantile(0.75)


def iqr(dataframe, feature):
    return percentile75(dataframe, feature) - percentile25(dataframe, feature)


def upper_limit(dataframe, feature, mult=1):
    return percentile75(dataframe, feature) + 1.5 * iqr(dataframe, feature) * mult


def lower_limit(dataframe, feature, mult=1):
    return percentile25(dataframe, feature) - 1.5 * iqr(dataframe, feature) * mult


def normal_lower_limit(dataframe, feature, mult=1):
    return dataframe[feature].agg("mean") - 3 * np.std(dataframe[feature]) * mult


def normal_upper_limit(dataframe, feature, mult=1):
    return dataframe[feature].agg("mean") + 3 * np.std(dataframe[feature]) * mult


def outliers_above(dataframe, feature):
    return dataframe[dataframe[feature] > upper_limit(dataframe, feature)]


def outliers_below(dataframe, feature):
    return dataframe[dataframe[feature] < lower_limit(dataframe, feature)]


def get_global_outliers(dataframe, feature):
    return dataframe[dataframe[feature] > upper_limit(dataframe, feature)].append(
        dataframe[dataframe[feature] < lower_limit(dataframe, feature)])


def cap_outliers(data, train, feature, strategy="Normal"):
    # Cap the "feature" column in "data" dataset according to the statistic in the same column in "train" dataset

    # If the percentage of the outliers in the feature is larger than a threshold, add a multiplier to broaden the
    # capping limits, to achieve less aggressive capping.
    threshold_percentile = 0.05
    multiplier = 1
    if len(get_global_outliers(train, feature)) / len(train) \
            > threshold_percentile:
        multiplier = 2
    if strategy == "Normal":
        data[feature].clip(
            normal_lower_limit(train, feature, multiplier),
            normal_upper_limit(train, feature, multiplier),
            inplace=True)
    else:
        data[feature].clip(
            lower_limit(train, feature, multiplier),
            upper_limit(train, feature, multiplier),
            inplace=True)


def cap_all_features_in_dataframe(data, train):
    # Iterate over all features and cap them according to their corresponding technique

    normal_features = ["sugar_levels",  "PCR_06", "PCR_07"]
    skewed_features = ["age", "num_of_siblings", "conversations_per_day", "PCR_03",
                       "sport_activity", "PCR_01", "PCR_02", "PCR_04", "PCR_08", "PCR_10"]

    for feature in normal_features:
        cap_outliers(data, train, feature, "Normal")
    for feature in skewed_features:
        cap_outliers(data, train, feature, "Skewed")

    return data, train


def cap_feature_with_respect_to_another(data, train, column_to_cap, intervals_column, number_of_intervals):
    """
    Divide the data in "column_to_cap" into intervals with respect to "intervals_column" column. Cap each group
    separately
    """
    interval_groups = get_interval_groups(train, intervals_column, number_of_intervals)
    # The series where the cleaned column will be stored
    clean_column = pd.Series()

    # Iterate over all weight groups and cap the data
    for i in range(1, len(interval_groups)):
        # Make sure that all values of data are included in a group, even if min value of data is smaller than that
        # of train, and similarly for max
        if i == 1:
            clean_group = data[(data[intervals_column] <= interval_groups[i])].copy()
        elif i == (len(interval_groups) - 1):
            clean_group = data[(data[intervals_column] > interval_groups[i - 1])].copy()
        else:
            clean_group = data[(data[intervals_column] > interval_groups[i - 1]) &
                               (data[intervals_column] <= interval_groups[i])].copy()

        train_group = train[(train[intervals_column] > interval_groups[i - 1]) &
                            (train[intervals_column] <= interval_groups[i])].copy()
        cap_outliers(clean_group, train_group, column_to_cap)
        clean_column = pd.concat([clean_column, clean_group[column_to_cap]])

    data.drop(column_to_cap, inplace=True, axis=1)
    # Update the dataset with the capped values
    data[column_to_cap] = clean_column

    return data


def remove_problematic_pcr_week(df):
    # As we have seen, most of the outliers in the PCR tests columns come from this specific week, leading to believe
    # that the tests from this week are erroneous. We nullify all the tests in this week, and later will impute to get
    # healthier values
    faulty_start = datetime.date.fromisoformat("2020-05-26").toordinal()
    faulty_end = datetime.date.fromisoformat("2020-06-05").toordinal()

    pcr_i = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_06',
             'PCR_07', 'PCR_08', 'PCR_10']

    pcr_mask = df["pcr_date"].between(
        faulty_start, faulty_end).apply(lambda x: not x)

    for pcr in pcr_i:
        df[pcr] = df[pcr_mask][pcr]

    return df


########################################################


def set_columns(df, bloodtype_to_binary):
    """
    Make new columns with relevant data from the existing columns, and covert data into more convenient formats to work
    with
    :param df: Dataset to set
    :return: Set dataset
    """
    # Turn sex column to binary
    df["sex"] = df["sex"].apply(lambda x: 1 if (x == "F") else -1)

    # Get the zipcode from address
    df["zip_code"] = address_to_zipcode(df["address"])

    # Get manageable data from the symptoms vector by separating it to several columns where
    # each column is its own symptom. 1 signifies that the subject has the symptom, 0 that the
    # subject doesn't have it

    symptoms_lists = df["symptoms"].str.split(';').dropna()
    symptom_vector = [symptom for symptoms in symptoms_lists for symptom in symptoms]
    symptoms_list = list(set(symptom_vector))

    for symptom in symptoms_list:
        df[symptom] = symptoms_lists.apply(lambda x: int(symptom in x))

    # Impute the blood type column, and then do OHE to get a column for each blood type
    impute_feature(df, "blood_type", "Sample")
    if bloodtype_to_binary:
        df = pd.concat([df.drop(["blood_type"], axis=1),
                        df["blood_type"].apply(lambda x: 1 if 'A' in x else 0)],
                        axis=1)

    # Turn pcr date to ordinal
    df["pcr_date"] = df["pcr_date"].astype("datetime64")
    df["pcr_date"] = df["pcr_date"].dropna().apply(lambda x: x.toordinal())

    # Remove all PCR tests from the problematic week (as described in question 13 in the dry part)
    remove_problematic_pcr_week(df)

    return df


def clean_outliers(data, train):
    # Cap local age and sugar levels column outliers according to the weight column
    # we arbitrarily decided to divide into 10 weight groups, yet this parameter can be toggled, to get better results.
    cap_feature_with_respect_to_another(data, train, 'age', 'weight', 10)
    cap_feature_with_respect_to_another(data, train, 'sugar_levels', 'weight', 10)

    # Cap all features
    data, train = cap_all_features_in_dataframe(data, train)

    return data, train


def impute_data(data):
    """
    Impute all columns in dataset with different imputation techniques
    :param data: dataset to impute
    :return: Imputed dataset
    """
    median_impute = ['num_of_siblings', 'conversations_per_day', 'sport_activity', 'household_income']
    mean_impute = ['PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_06', 'PCR_07',
                   'PCR_08', 'PCR_10']
    binary_impute = ['shortness_of_breath', 'headache', 'cough', 'low_appetite', 'fever']
    sample_impute = ['weight', 'sex', 'zip_code']
    by_weight_impute = ['age', 'sugar_levels']

    for feature in median_impute:
        data = impute_feature(data, feature, "Median")
    for feature in mean_impute:
        data = impute_feature(data, feature, "Mean")
    for feature in binary_impute:
        data = impute_feature(data, feature, "Binary")
    for feature in sample_impute:
        data = impute_feature(data, feature, "Sample")

    # The following features are imputed by dividing the dataset into weight groups and imputing correspondingly
    # to maintain the correlation between these features and the weight feature, and still use sample impute which
    # better fits these features
    for feature in by_weight_impute:
        data = impute_feature(data, feature, "by_weight")

    return data


def remove_unnecessary_features(df, remove_targets):
    """
    After we have gotten all the data we need from these columns, drop them.
    :param df: The dataset whose superfluous columns we want to get rid of
    :return: The clean dataset
    """
    columns_to_drop = ["symptoms", "job", "current_location", "patient_id", "weight",
                       "address", "happiness_score", "pcr_date", "PCR_05", "PCR_09"]

    if remove_targets:
        columns_to_drop += ["VirusScore"]

    for column in columns_to_drop:
        df.drop(column, inplace=True, axis=1)

    return df


def normalize_columns(columns, method="standard"):
    if method == "standard":
        standard_scaler = StandardScaler()
        standard_scaler.fit(columns)
        scaled_columns = standard_scaler.transform(columns)
    else:
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(columns)
        scaled_columns = min_max_scaler.transform(columns)

    return scaled_columns


def normalize(df):
    standard_columns = ["PCR_06", "PCR_03", "PCR_07", "PCR_10", "sugar_levels"]
    min_max_columns = ["PCR_01", "PCR_02", "PCR_04", "PCR_08",  "household_income",
                       "age", "num_of_siblings", "conversations_per_day", "sport_activity", "zip_code"]

    df[standard_columns] = normalize_columns(df[standard_columns], "standard")

    df[min_max_columns] = normalize_columns(df[min_max_columns], "min_max")

    return df


def prepare_data(data, training_data, normalize_columns=1, remove_targets=1, bloodtype_to_binary=True):
    # Generate random seed to replicate results
    np.random.seed(11)

    # Get a copy of the data so we don't spoil the original data
    d = data.copy()
    t = training_data.copy()

    # Part 1 - Prepare columns to manage the analysis operations
    d = set_columns(d, bloodtype_to_binary)
    t = set_columns(t, bloodtype_to_binary)

    # Part 2 - Clean outliers from the data
    d, t = clean_outliers(d, t)

    # Part 3 - Impute missing data
    d = impute_data(d)

    # Separate target features from data
    target_features = pd.Series(d["VirusScore"])

    # Part 4 - Remove Unnecessary features
    d = remove_unnecessary_features(d, remove_targets)

    d = normalize(d) if normalize_columns == 1 else d

    return d, target_features