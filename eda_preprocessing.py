import pandas as pd
import os

import config
from utils.utils import DataPreprocessor, FeatureExtractor, DataModelUtils
from utils.utils_plot import Plotter

# Ensure the data directory exists
if not os.path.exists(config.data_dir):
    os.makedirs(config.data_dir)

dmutils=DataModelUtils()

# Download and unzip the dataset
dmutils.download_and_unzip(config.url, config.zip_path, config.data_dir)

df = pd.read_csv(config.file_path_hhps, delimiter=';')

print(df.head())
print(df.shape)
print(df.info())

# Initialize the preprocessor with raw data
preprocessor = DataPreprocessor(df)

# Apply various preprocessing steps
preprocessor.merge_date_time()
preprocessor.convert_to_numeric()
preprocessor.convert_to_wattmin()
preprocessor.compute_other_active_power()
preprocessor.drop_columns(config.cols_to_drop)
preprocessor.handle_missing_values(strategy="ffill")
preprocessor.rename_columns(config.cols_to_rename)

# To get the preprocessed DataFrame
preprocessed_df = preprocessor.get_preprocessed_data()

print(preprocessed_df.head())

df_proc = preprocessed_df.copy()

feature_extractor = FeatureExtractor(df_proc)

dfj = feature_extractor.extract_features()

print(dfj.head())

dfj[config.col_name_smk +'_ratio'] = feature_extractor.compute_ratio(config.col_name_smk, config.col_name_tpc)
dfj[config.col_name_sml +'_ratio'] = feature_extractor.compute_ratio(config.col_name_sml, config.col_name_tpc)
dfj[config.col_name_smwa +'_ratio'] = feature_extractor.compute_ratio(config.col_name_smwa, config.col_name_tpc)
dfj[config.col_name_smo +'_ratio'] = feature_extractor.compute_ratio(config.col_name_smo, config.col_name_tpc)


# Initialize the preprocessor with raw data
preprocessor1 = DataPreprocessor(dfj)

# Apply various preprocessing steps
preprocessor1.handle_missing_values(strategy="bfill")


dfj.to_csv(config.file_path_prep, index=False)

print(dfj.head(10))

# Sort the DataFrame by Date_time
dfj = dfj.sort_values(by='Date_time')


plotter = Plotter(dfj)

plotter.plot_power_consumption()

plotter.plot_hourly_power_consumption()

plotter.plot_hourly_daily_mean_consumption()

plotter.plot_hist_daily_power_distribution()

plotter.plot_power_consumption_with_dip()

plotter.plot_seasonal_power_consumption()

plotter.plot_power_consumption_by_submeter()

plotter.plot_weekend_weekday_power_consumption()

plotter.plot_pivot_table_heatmap()

plotter.plot_correlation_heatmap()
