




# URL of the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'

# Path where the zip file will be saved temporarily
zip_path = 'household_power_consumption.zip'

# Directory to extract the contents to
data_dir = 'data'


file_path_hhps ='./data/household_power_consumption.txt'


cols_to_drop ={'index', 'Date', 'Time', 'Global_reactive_power', 'Voltage', 'Global_intensity'}

cols_to_rename ={'Global_active_power': 'Total_Power_Consumption', 
           'Sub_metering_1': 'Submeter_kitchen',
           'Sub_metering_2': 'Submeter_laundry',
           'Sub_metering_3': 'Submeter_waterheat_aircon',
           'Other_active_power': 'Submeter_others'}

col_name_smk= 'Submeter_kitchen'
col_name_sml= 'Submeter_laundry'
col_name_smwa= 'Submeter_waterheat_aircon'
col_name_smo= 'Submeter_others'
col_name_tpc= 'Total_Power_Consumption'

file_path_prep = './data/preprocessed_df.csv'


columns_name_dt =['Date_time']



model_name_xgb='xgboost'
param_xgb={
    'n_estimators': 300, 
    'objective': 'reg:squarederror',
    'base_score': 0.5,
    'booster': 'gbtree',    
    'max_depth': 6,
    'random_state': 42
    }

model_name_rf='randomforest'
param_rf = {
    'n_estimators':300, 
    'random_state':42, 
    'max_depth': None
    }

model_name_dt = 'decisiontree'
param_dt = {
    'min_samples_leaf':2,     
    'random_state':0
    }

model_name_ab='adaboost'
param_ab = { 
    'n_estimators':100,
    'learning_rate': 0.1
}

# Define the parameter grid
param_grid = { 
    'n_estimators': [300], 
    'random_state':[42],
    'max_depth': [None, 3, 5], 
    'min_samples_leaf': [3, 5, 7],
    'min_samples_split': [3, 5, 10]
} 

model_names = ['XGBoost', 'Random Forest', 'Decision tree', 'AdaBoost']

file_path_reco_test = './data/df_recommend_test.csv'


file_path_reco_entire = './data/df_recommend_entire.csv'


columns_name_to_drop ={'hour', 'day', 'month', 'quarter', 'is_weekend', 'is_summer', 'Submeter_kitchen_ratio',
                       'Submeter_laundry_ratio', 'Submeter_waterheat_aircon_ratio', 'Submeter_others_ratio'}