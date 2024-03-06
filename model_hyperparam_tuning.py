import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import config

from utils.utils import DataPreprocessor, TimeseriesDataPreprocessor
from utils.model import ConsumptionModelHyperParam



df = pd.read_csv(config.file_path_prep)

# Ensure Date_time is in Datetime Format
df['Date_time'] = pd.to_datetime(df['Date_time'])

# Sort the DataFrame by Date_time
df = df.sort_values(by='Date_time')

ts_preprocess=TimeseriesDataPreprocessor(df)

start_date, end_date= ts_preprocess.get_start_end_date()

df_proc= ts_preprocess.get_filtered_dataframe()

ts_preprocess1=TimeseriesDataPreprocessor(df_proc)

train_data, test_data = ts_preprocess1.split_timeseries_data()

test_start_date = test_data['Date_time'].iloc[0]

preprocessor1 = DataPreprocessor(train_data.copy())
preprocessor1.drop_columns(config.columns_name_dt)
train_df = preprocessor1.get_preprocessed_data()

print(train_df.info())

preprocessor2 = DataPreprocessor(test_data.copy())
preprocessor2.drop_columns(config.columns_name_dt)
test_df = preprocessor2.get_preprocessed_data()

print(test_df.info())

features= train_df.copy()

print(features.columns)

# Assuming 'Total_Power_Consumption' is the target variable
target = features.pop('Total_Power_Consumption')

print(target)

print(features.columns)


# Initialize the model
base_estimator = RandomForestRegressor()
consumption_model = ConsumptionModelHyperParam(features, target, base_estimator)

print('Model training is in progress.')

# Train the model with hyperparameter tuning
consumption_model.train(config.param_grid)






