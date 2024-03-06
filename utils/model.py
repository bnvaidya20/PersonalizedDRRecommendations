
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class ModelSelector:
    def __init__(self):
        self.model = None

    def select_regressor(self, model_name, params={}):
        if model_name.lower() == 'randomforest':  
            self.model = RandomForestRegressor(**params)
        
        elif model_name.lower() == 'decisiontree':
            self.model = DecisionTreeRegressor(**params)

        elif model_name.lower() == 'adaboost':
            self.model = AdaBoostRegressor(**params)

        elif model_name.lower() == 'xgboost':
            self.model = xgb.XGBRegressor(**params)

        else:
            raise ValueError(f"Model '{model_name}' is not supported.")
        
        return self.model

class ConsumptionModelEx:

    def __init__(self, features, target, model_name=None, model_params={}):
        """
        Initialize the ConsumptionModel with features, target, and no default model.
        
        :param features: DataFrame or array-like, features for training the model
        :param target: Series or array-like, target variable for training the model
        """
        self.features = features
        self.target = target
        self.selector = ModelSelector()  # Create an instance of ModelSelector
        self.model = self.selector.select_regressor(model_name, model_params)  # Select and initialize the model

    def train(self):
        """
        Train the selected model using the provided features and target.
        Splits the data into training and testing sets, fits the model on the training data,
        makes predictions on the test set, and prints various evaluation metrics.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)

        
        # Make predictions on the testing set
        predictions = self.model.predict(X_test)
        
        # Calculate and print the MSE of the predictions
        mse = mean_squared_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)

        print(f"""
            Evaluation metrics Model {self.model} trained with 
            MSE: {mse}
            RMSE: {rmse}
            MAE: {mae}
        """)

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        :param X: DataFrame or array-like, features for making predictions
        :return: array, predicted values
        """
        return self.model.predict(X)
    

class ConsumptionModelEx:

    def __init__(self, features, target, model_name=None, model_params={}):
        """
        Initialize the ConsumptionModel with features, target, and no default model.
        
        :param features: DataFrame or array-like, features for training the model
        :param target: Series or array-like, target variable for training the model
        """
        self.features = features
        self.target = target
        self.selector = ModelSelector()  # Create an instance of ModelSelector
        self.model = self.selector.select_regressor(model_name, model_params)  # Select and initialize the model
        # Define your preprocessing steps and model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  
            ('model', self.model)
        ])

    def train(self):
        """
        Train the selected model using the provided features and target.
        Splits the data into training and testing sets, fits the model on the training data,
        makes predictions on the test set, and prints various evaluation metrics.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        
        # self.model.fit(X_train, y_train)
        # Train the pipeline on your training data (excluding the target variable)
        self.pipeline.fit(X_train, y_train)

        
        # Make predictions on the testing set
        # predictions = self.model.predict(X_test)
        predictions = self.pipeline.predict(X_test)

        
        # Calculate and print the MSE of the predictions
        mse = mean_squared_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)

        print(f"""
            Evaluation metrics Model {self.model} trained with 
            MSE: {mse}
            RMSE: {rmse}
            MAE: {mae}
        """)

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        :param X: DataFrame or array-like, features for making predictions
        :return: array, predicted values
        """
        # return self.model.predict(X)
        return self.pipeline.predict(X)


class ModelUtils:

    def split_data(self, train_data):
        """Split the training data into training and validation sets."""
        features = train_data.copy()
        target = features.pop('Total_Power_Consumption')

        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

        return X_train, X_val, y_train, y_val

    def split_test_data(self, test_data):
        """Split the test data into features and target."""
        test_data = test_data.copy()
        X_test = test_data.drop(columns=['Total_Power_Consumption'])
        y_test = test_data['Total_Power_Consumption']

        return X_test, y_test    


class Model:
    def __init__(self, model_name=None, model_params={}):
        """Initialize the Model with a specific regressor and parameters."""
        self.selector = ModelSelector()  
        self.model = self.selector.select_regressor(model_name, model_params)
        self.model_name = model_name.lower()

    def train(self, X_train, X_val, y_train, y_val):

        """Train the model on the training data."""

        fit_methods = {
            'randomforest': self.model.fit,
            'decisiontree': self.model.fit,
            'adaboost': self.model.fit,
            'xgboost': lambda X, y: self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=500)
        }

        if self.model_name in fit_methods:
            fit_methods[self.model_name](X_train, y_train)
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")
        
        return self.model

    def predict(self, X_true, y_true, model):

        y_pred = model.predict(X_true)

        # Calculate the metrics of the predictions
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"""
            Evaluation metrics:
            MSE: {mse}
            RMSE: {rmse}
            MAE: {mae}
            R2: {r2}
        """)
        return y_pred, mse, rmse, mae, r2


class ConsumptionModelHyperParam:
    def __init__(self, features, target, model):
        self.features = features
        self.target = target
        self.model = model
        # Define a pipeline combining a standard scaler and the model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model)
        ])
        
    def train(self, param_grid, cv=5, scoring='neg_mean_squared_error'):
        """
        Train the model with hyperparameter tuning using GridSearchCV.

        :param param_grid: Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
        :param cv: Number of cross-validation folds.
        :param scoring: A single string to evaluate the predictions on the test set.
        """
        # Ensure model hyperparameters are prefixed with 'model__' for the pipeline
        param_grid_prefixed = {'model__' + key: value for key, value in param_grid.items()}
        
        grid_search = GridSearchCV(self.pipeline, param_grid_prefixed, cv=cv, scoring=scoring, return_train_score=True)
        grid_search.fit(self.features, self.target)

        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", -grid_search.best_score_)

        # Update the pipeline model with the best found parameters
        self.pipeline.set_params(**grid_search.best_params_)

    def predict(self, X):
        return self.pipeline.predict(X)


