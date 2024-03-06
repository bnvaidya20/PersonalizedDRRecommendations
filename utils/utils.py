
import pandas as pd
import numpy as np

import requests
import zipfile
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class DataPreprocessor:

    def __init__(self, data):
        self.data = data

    def merge_date_time(self):
        self.data['Date_time'] = pd.to_datetime(self.data['Date'] + ' ' + self.data['Time'])


    def convert_to_numeric(self):
        self.data['Global_active_power'] = pd.to_numeric(self.data['Global_active_power'], errors='coerce')
        self.data['Sub_metering_1'] = pd.to_numeric(self.data['Sub_metering_1'], errors='coerce')
        self.data['Sub_metering_2'] = pd.to_numeric(self.data['Sub_metering_2'], errors='coerce')
        self.data['Sub_metering_3'] = pd.to_numeric(self.data['Sub_metering_3'], errors='coerce')

    def convert_to_wattmin(self):
        self.data['Global_active_power'] = self.data['Global_active_power'] * 1000 / 60

    def compute_other_active_power(self):

        self.data['Other_active_power'] = self.data['Global_active_power'] - (self.data['Sub_metering_1'] + self.data['Sub_metering_2'] + self.data['Sub_metering_3'])

    def handle_missing_values(self, strategy="ffill"):
        """
        Handle missing values in the dataset.
        strategy: "mean", "median", "drop", "ffill", "bfill" - method for handling missing values
        """
        # Handle missing values
        print(f"""
        Missing Value found: 
        {self.data.isna().sum()}
        """)
        print('*'*50)

        if strategy == "mean":
            # Replace missing values with the mean
            self.data.fillna(self.data.mean(), inplace=True)
            print('Fill na with values completed')

        elif strategy == "median":
            # Replace missing values with the median
            self.data.fillna(self.data.median(), inplace=True)
            print('Fill na with values completed')

        elif strategy == "ffill":
            # Fill na with values forward
            self.data.fillna(method='ffill', inplace=True)
            print('Fill na with values forward completed')

        elif strategy == "bfill":
            # Fill na with values backward
            self.data.fillna(method='bfill', inplace=True)
            print('Fill na with values backward completed')

        elif strategy == "drop":
            # Drop rows with missing values
            self.data.dropna(inplace=True)
            print('Fill na with values completed')

    
    def drop_columns(self, columns_name):
       """
       Drop columns that are not needed for analysis.
       columns_name: list of columns to be dropped
       """
       self.data = self.data.drop(columns=columns_name, errors='ignore')
    
    def rename_columns(self, cols_name):
        """
            Rename columns that are needed for analysis.
            cols_name: list of columns to be renamed
        """

        self.data.rename(columns=cols_name, inplace=True)

    def get_preprocessed_data(self):
        """
        Return the preprocessed data.
        """

        return self.data
    

class FeatureExtractor:
    def __init__(self, data):
        self.data = data

    def extract_features(self):
        # Extract time-based features like hour of day, day of week, etc.
        self.data['hour'] = self.data['Date_time'].dt.hour
        self.data['day'] = self.data['Date_time'].dt.dayofweek
        self.data['month'] = self.data['Date_time'].dt.month
        self.data['quarter'] = self.data['Date_time'].dt.quarter

        self.data['is_weekend'] = self.data['day'].apply(lambda x: 1 if x >= 5 else 0)
        self.data['is_summer'] = self.data['month'].apply(lambda x: 1 if x in [5, 6, 7, 8, 9, 10] else 0)

        self.data['day'] = self.data['day'] + 1

        return self.data

    def compute_ratio(self, col_name, col_namef):
        self.data[col_name +'_ratio']= self.data[col_name] / self.data[col_namef]

        return self.data[col_name +'_ratio']


       
class TimeseriesDataPreprocessor:

    def __init__(self, data):
        self.data = data.copy()

    def get_start_end_date(self):

        # Filter the DataFrame Based on Start and End Dates
        self.start_date = self.data['Date_time'].min()
        self.end_date = self.data['Date_time'].max()
        
        return self.start_date, self.end_date
    
    def get_filtered_dataframe(self):
        # Filter the DataFrame
        self.data = self.data[(self.data['Date_time'] >= self.start_date) & (self.data['Date_time'] <= self.end_date)]

        # print("Columns in DataFrame:", self.data.columns.tolist())

        return self.data 
    
    def split_timeseries_data(self):

        # Calculate the Split Index
        split_index = int(len(self.data) * 0.7)

        # Split the Data into Training and Testing Sets

        # Training data: The first 80% of the filtered DataFrame
        train_data = self.data.iloc[:split_index]

        # print(train_data.shape)

        # Testing data: The remaining 20% of the filtered DataFrame
        test_data = self.data.iloc[split_index:]

        # print(test_data.shape)

        return train_data, test_data
    


class DataModelUtils:

    def download_and_unzip(self, url, zip_path, extract_dir):
        # Download the file from the specified url
        print(f"Downloading {url}")
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to a zip file
            with open(zip_path, 'wb') as file:
                file.write(response.content)
            print("Download completed.")
            
            # Unzip the file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted to {extract_dir}")
            
            # Remove the zip file after extraction
            os.remove(zip_path)
            print("Zip file removed.")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")


    def get_feature_importance(self, model, feature_names):
        importances = model.feature_importances_
        print(importances)
        for i, j in enumerate(importances):
            # Use 'feature_names' list for feature names instead of 'model.feature_names_in_'
            print(f'Feature: {feature_names[i]}, Score: {j:.5f}')
        return importances

    def get_data_with_feature_importance(self, X_train, X_val, X_test, importances):
        # Select the most important features
        num_features = 6
        fe_sorted_idx = np.argsort(importances)[-num_features:]

        # Restrict dataset to these features
        X_train_sel = X_train[:, fe_sorted_idx[-num_features:]]  
        X_val_sel = X_val[:, fe_sorted_idx[-num_features:]]
        X_test_sel = X_test[:, fe_sorted_idx[-num_features:]]

        return X_train_sel, X_val_sel, X_test_sel

    @staticmethod
    def compare_metrics_org_fs(mse_ts, rmse_ts, mae_ts, r2_ts, mse_ts_sel, rmse_ts_sel, mae_ts_sel, r2_ts_sel):
        # Compare the Results
        change_mse = mse_ts - mse_ts_sel
        change_rmse = rmse_ts - rmse_ts_sel
        change_mae = mae_ts - mae_ts_sel
        change_r2 = r2_ts - r2_ts_sel

        print(f"""Changes in metrics:
            MSE: {change_mse}
            RMSE: {change_rmse}
            MAE: {change_mae}
            R2: {change_r2}
        """)

    @staticmethod
    def get_test_final(test_df, y_test_pred):
        test_df_final = test_df.copy()
        test_df_final['Prediction'] = y_test_pred
        return test_df_final

    @staticmethod
    def compute_rmse_score(test_df_final):
        # RMSE Score
        score = mean_squared_error(test_df_final['Total_Power_Consumption'], test_df_final['Prediction'], squared=False)
        return score

    @staticmethod
    def compare_metrics(models, model_names, X_test, y_test):
        metrics_dict = {'Model': [], 'MSE': [], 'RMSE': [], 'MAE': [], 'R2': []}

        for model, name in zip(models, model_names):
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics_dict['Model'].append(name)
            metrics_dict['MSE'].append(mse)
            metrics_dict['RMSE'].append(rmse)
            metrics_dict['MAE'].append(mae)
            metrics_dict['R2'].append(r2)
           
        metrics_df = pd.DataFrame(metrics_dict)

        return metrics_df


class DataScaler:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    

    def fit_transform_train(self, X_train, y_train):
            """Fit to training data and transform both X and y."""
            # Assuming X_train is a DataFrame and you want to scale only numeric columns
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            
            # Convert y_train to a numpy array and reshape
            y_train_scaled = self.scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
            return X_train_scaled, y_train_scaled
    
    def transform_test(self, X_test, y_test):
        """Transform test data using the scalers fitted on the training data."""
        X_test_scaled = self.scaler_X.transform(X_test)
        # Convert y_test to a numpy array and reshape
        y_test_scaled = self.scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).flatten()
        return X_test_scaled, y_test_scaled
    
    def inverse_transform_y(self, y_scaled):
        """Inverse transform the scaled y values back to the original scale."""
        # Reshape y_scaled to 2D if it's a 1D array before inverse transformation
        y_original = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return y_original