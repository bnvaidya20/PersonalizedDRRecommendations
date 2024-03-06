import pandas as pd

import config

from utils.utils import DataPreprocessor, TimeseriesDataPreprocessor
from utils.model import ConsumptionModelEx
from utils.recommender_scheme import AdvancedDRRecommender

df = pd.read_csv(config.file_path_prep)

print(df.head())

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

print(train_df.head())

preprocessor2 = DataPreprocessor(test_data.copy())
preprocessor2.drop_columns(config.columns_name_dt)
test_df = preprocessor2.get_preprocessed_data()

print(test_df.head())


features= train_df.copy()

print(features.columns)

# Assuming 'Total_Power_Consumption' is the target variable
target = features.pop('Total_Power_Consumption')

print(target)

print(features.columns)


print('Processing Model Training... ')

# Initialize ConsumptionModel with a specific model and parameters
consumption_model = ConsumptionModelEx(features, target, model_name=config.model_name_rf, 
                                      model_params=config.param_rf)

# Train the model
consumption_model.train()


print('Computing Recommendations..')


adrecommender = AdvancedDRRecommender(consumption_model, test_df)

recommendations = adrecommender.generate_recommendations()

print(recommendations)

dtest_df = test_df.copy()

dtest_df['Recommendation']= recommendations

print(dtest_df.head())

# dtest_df.to_csv(config.file_path_reco_test, index=False)


# preprocessor3 = DataPreprocessor(df_proc.copy())
# preprocessor3.drop_columns(columns_name)
# processed_df = preprocessor3.get_preprocessed_data()

# adrecommender = AdvancedDRRecommender(consumption_model_rf, processed_df)

# recommendations = adrecommender.generate_recommendations()

# print(recommendations)

# proc_df = processed_df.copy()

# proc_df['Recommendation']= recommendations

# print(proc_df.head())

# proc_df.to_csv(config.file_path_reco_entire, index=False)




