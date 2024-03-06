import pandas as pd

import config

from utils.utils import DataPreprocessor, TimeseriesDataPreprocessor, DataModelUtils, DataScaler
from utils.utils_plot import PredPlotter, BasicPlotter
from utils.model import Model, ModelUtils


df = pd.read_csv(config.file_path_prep)

print(df.head(10))

# Ensure Date_time is in Datetime Format
df['Date_time'] = pd.to_datetime(df['Date_time'])

# Sort the DataFrame by Date_time
df = df.sort_values(by='Date_time')

ts_preprocess=TimeseriesDataPreprocessor(df)

start_date, end_date= ts_preprocess.get_start_end_date()

print(start_date, end_date)

df_proc= ts_preprocess.get_filtered_dataframe()

print(df_proc.describe().T)

ts_preprocess1=TimeseriesDataPreprocessor(df_proc)

train_data, test_data = ts_preprocess1.split_timeseries_data()

test_start_date = test_data['Date_time'].iloc[0]

basplotter=BasicPlotter()

basplotter.plot_power_consumption_train_test(train_data, test_data, test_start_date)

preprocessor1 = DataPreprocessor(train_data.copy())
preprocessor1.drop_columns(config.columns_name_dt)
train_df = preprocessor1.get_preprocessed_data()

print(train_df.info())

preprocessor2 = DataPreprocessor(test_data.copy())
preprocessor2.drop_columns(config.columns_name_dt)
test_df = preprocessor2.get_preprocessed_data()

print(test_df.info())

mutils= ModelUtils()

X_train, X_val, y_train, y_val = mutils.split_data(train_df)

print(X_train.shape)
print(X_val.shape)

X_test, y_test = mutils.split_test_data(test_df)

print(X_test.shape)
                

# Initialize the DataScaler
data_scaler = DataScaler()

# Scale the training data
X_train_scaled, y_train_scaled = data_scaler.fit_transform_train(X_train, y_train)

# Scale the val data
X_val_scaled, y_val_scaled = data_scaler.transform_test(X_val, y_val)

# Scale the test data
X_test_scaled, y_test_scaled = data_scaler.transform_test(X_test, y_test)

print(X_train_scaled.shape)
print(X_val_scaled.shape)
print(X_test_scaled.shape)


g_model1 = Model(model_name=config.model_name_xgb, model_params=config.param_xgb)
g_model2 = Model(model_name=config.model_name_rf, model_params=config.param_rf)
g_model3 = Model(model_name=config.model_name_dt, model_params=config.param_dt)
g_model4 = Model(model_name=config.model_name_ab, model_params=config.param_ab)


# Train the model
print("Model training is in progress ...")

xgb_model = g_model1.train(X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled)

rf_model = g_model2.train(X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled)

dt_model = g_model3.train(X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled)

ab_model = g_model4.train(X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled)

print("Model training completed.")

# y_val_pred_scaled,_,_,_,_ = g_model1.predict(X_val_scaled, y_val_scaled, xgb_model)
y_val_pred_scaled,_,_,_,_ = g_model2.predict(X_val_scaled, y_val_scaled, rf_model)
# y_val_pred_scaled,_,_,_,_ = g_model3.predict(X_val_scaled, y_val_scaled, dt_model)
# y_val_pred_scaled,_,_,_,_ = g_model4.predict(X_val_scaled, y_val_scaled, ab_model)

# y_test_pred_scaled, mse_ts,rmse_ts,mae_ts,r2_ts = g_model1.predict(X_test_scaled, y_test_scaled, xgb_model)
y_test_pred_scaled, mse_ts,rmse_ts,mae_ts,r2_ts = g_model2.predict(X_test_scaled, y_test_scaled, rf_model)
# y_test_pred_scaled, mse_ts,rmse_ts,mae_ts,r2_ts = g_model3.predict(X_test_scaled, y_test_scaled, dt_model)
# y_test_pred_scaled, mse_ts,rmse_ts,mae_ts,r2_ts = g_model4.predict(X_test_scaled, y_test_scaled, ab_model)

# Inverse transform the predictions back to the original scale
y_test_pred = data_scaler.inverse_transform_y(y_test_pred_scaled)

basplotter.plot_actual_pred(y_test, y_test_pred)


models = [xgb_model, rf_model, dt_model, ab_model]

utils=DataModelUtils()

metrics_df = utils.compare_metrics(models, config.model_names, X_test_scaled, y_test_scaled)

print(metrics_df)

test_df_final = utils.get_test_final(test_df, y_test_pred)

score = utils.compute_rmse_score(test_df_final)

print(f'RMSE Score on Test set: {score:0.4f}')


# Built-in Feature Importance

feature_names = X_train.columns  # Get the feature names from your DataFrame

importances = utils.get_feature_importance(rf_model, feature_names)

# Plot feature importances
basplotter.plot_feature_importances(rf_model, feature_names)

X_train_fi, X_val_fi,X_test_fi = utils.get_data_with_feature_importance(X_train_scaled, X_val_scaled, X_test_scaled, importances)

print(X_train_fi.shape)
print(X_val_fi.shape)
print(X_test_fi.shape)

# Train a new model on the reduced dataset
print("Model training is in progress ...")
rf_model_fi = g_model2.train(X_train_fi, X_val_fi, y_train_scaled, y_val_scaled)

print("Model training completed and saved.")

# Evaluate the new model on val data
y_val_pred_fi,_,_,_,_ = g_model2.predict(X_val_fi, y_val_scaled, rf_model_fi)

# Evaluate the new model on test data
y_test_pred_fi, mse_ts_fi, rmse_ts_fi, mae_ts_fi, r2_ts_fi = g_model2.predict(X_test_fi, y_test_scaled, rf_model_fi)

utils.compare_metrics_org_fs(mse_ts, rmse_ts, mae_ts, r2_ts, mse_ts_fi, rmse_ts_fi, mae_ts_fi, r2_ts_fi)


# Inverse transform the predictions back to the original scale
y_test_pred_fi_inv = data_scaler.inverse_transform_y(y_test_pred_fi)

test_df_final_fi = utils.get_test_final(test_df, y_test_pred_fi_inv)


basplotter.plot_actual_pred(y_test, y_test_pred_fi_inv)

predplotter = PredPlotter(df_proc, test_df_final_fi)

predplotter.plot_original_prediction()

predplotter.plot_prediction_test_period(test_start_date, end_date)

predplotter.plot_seasonal_prediction()









