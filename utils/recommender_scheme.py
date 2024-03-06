import pandas as pd


class AdvancedDRRecommender:
    def __init__(self, model, data):
        self.model = model
        self.data = data.copy()
        # self.recommendations = None

    def ensure_features(self):
        # Check for necessary columns in the DataFrame
        required_columns = ['Submeter_kitchen', 'Submeter_laundry', 'Submeter_waterheat_aircon', 'Submeter_others', 
                            'hour', 'is_weekend', 'is_summer']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            print(f"Missing columns in DataFrame: {missing_columns}")
            return None

    def generate_recommendations(self):

        self.ensure_features()

        # features=['Submeter_kitchen', 'Submeter_laundry', 'Submeter_waterheat_aircon',
        #                  'Submeter_others', 'hour', 'day', 'month', 'quarter', 'is_weekend',
        #                  'is_summer', 'Submeter_kitchen_ratio', 'Submeter_laundry_ratio',
        #                  'Submeter_waterheat_aircon_ratio', 'Submeter_others_ratio']
        test_features = self.data.drop(columns=['Total_Power_Consumption'], errors='ignore')

        # Predict the power consumption for each row in the dataset
        self.data['predicted_consumption'] = self.model.predict(test_features)

        # Define thresholds for high consumption
        high_consumption_threshold = self.data['predicted_consumption'].quantile(0.7)
        extreme_consumption_threshold = self.data['predicted_consumption'].quantile(0.9)

        # Define custom recommendation logic
        def custom_recommendations(row):
            recommendations = []

            # High overall consumption
            if row['predicted_consumption'] > extreme_consumption_threshold:
                recommendations.append("Consider reducing usage of all high-consumption appliances.")

            # Specific appliance usage recommendations
            if row['Submeter_kitchen'] > row[['Submeter_laundry', 'Submeter_waterheat_aircon', 'Submeter_others']].max():
                recommendations.append("Your kitchen appliances are consuming a lot. Consider using them during off-peak hours.")
            if row['Submeter_laundry'] > row[['Submeter_kitchen', 'Submeter_waterheat_aircon', 'Submeter_others']].max():
                recommendations.append("High laundry consumption detected. Try to consolidate laundry loads.")
            if row['Submeter_waterheat_aircon'] > row[['Submeter_kitchen', 'Submeter_laundry', 'Submeter_others']].max():
                recommendations.append("Your heating or cooling systems are drawing a lot of power. Check if settings are optimal.")

            # Time-based recommendations adjusted for TOU pricing, including summer schedule
            is_summer = row['is_summer']
            is_weekend = row['is_weekend']
            hour = row['hour']
            if is_summer:
                # Summer TOU pricing
                if not is_weekend and (11 <= hour < 17) and row['predicted_consumption'] > high_consumption_threshold:  # On-Peak hours on weekdays
                    recommendations.append("Consider reducing usage during weekday summer On-Peak hours (11 a.m. to 5 p.m.).")
                elif not is_weekend and ((7 <= hour < 11) or (17 <= hour < 19)) and row['predicted_consumption'] > high_consumption_threshold:  # Mid-Peak hours on weekdays
                    recommendations.append("Consider reducing usage during weekday summer Mid-Peak hours (7 a.m. to 11 a.m. and 5 p.m. to 7 p.m.).")
                elif (is_weekend or (hour < 7 or hour >= 19)) and row['predicted_consumption'] > high_consumption_threshold:  # Off-Peak hours
                    recommendations.append("Your consumption is higher during Off-Peak hours. This is generally okay, but you might still consider reducing usage if it's significantly high.")
            else:
                # Winter TOU pricing
                if not is_weekend and ((7 <= hour < 11) or (17 <= hour < 19)) and row['predicted_consumption'] > high_consumption_threshold:  # On-Peak hours on weekdays
                    recommendations.append("Consider reducing usage during weekday On-Peak hours (7 a.m. to 11 a.m. and 5 p.m. to 7 p.m.).")
                elif not is_weekend and (11 <= hour < 17) and row['predicted_consumption'] > high_consumption_threshold:  # Mid-Peak hours on weekdays
                    recommendations.append("Consider reducing usage during weekday Mid-Peak hours (11 a.m. to 5 p.m.).")
                elif (is_weekend or (hour < 7 or hour >= 19)) and row['predicted_consumption'] > high_consumption_threshold:  # Off-Peak hours
                    recommendations.append("Your consumption is higher during Off-Peak hours. This is generally okay, but you might still consider reducing usage if it's significantly high.")

            # Weekend-specific recommendations
            if row['is_weekend'] and row['predicted_consumption'] > high_consumption_threshold:
                recommendations.append("Weekend consumption is high. Consider outdoor activities.")

            return " | ".join(recommendations) if recommendations else "Your consumption is within normal range."

        # Apply custom recommendation logic to each row
        recommendations = self.data.apply(custom_recommendations, axis=1)

        return recommendations

class BasicDRRecommender:
    def __init__(self, model, data):
        self.model = model
        self.data = data.copy()

    def generate_recommendations(self):
        # Ensure the DataFrame contains all the necessary columns
        required_columns = ['Submeter_kitchen', 'Submeter_laundry', 'Submeter_waterheat_aircon',
                             'Submeter_others']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            print(f"Missing columns in DataFrame: {missing_columns}")
            return None

        # Assuming 'Total_Power_Consumption' is the target variable and should not be included in the features
        features = self.data.drop(columns=['Total_Power_Consumption'])

        if 'Total_Power_Consumption' in self.data.columns:
            # Predict the power consumption for each row in the dataset
            self.data['predicted_consumption'] = self.model.predict(features)
        else:
            print("'Total_Power_Consumption' column not found in DataFrame.")
            return None

        # Analyze the predictions to generate recommendations
        # For simplicity, let's assume a basic rule: if predicted consumption is above a threshold, recommend turning off non-essential appliances
        threshold = self.data['predicted_consumption'].quantile(0.7)
        recommendations = self.data.apply(lambda row: "Consider turning off non-essential appliances" if row['predicted_consumption'] > threshold else "Your consumption is within normal range", axis=1)
        return recommendations
    

class RecommendationQuery:
    def __init__(self, file_path):
        self.file_path = file_path
        self.recommendations = None

    def load_recommendations(self):
        try:
            self.recommendations = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")

    def get_recommendation(self, **query):
        if self.recommendations is None:
            print("No recommendations loaded. Loading recommendations now.")
            self.load_recommendations()

        if self.recommendations is not None:
            filtered_recommendations = self.recommendations.copy()
            for key, condition in query.items():
                if key in filtered_recommendations.columns:
                    if isinstance(condition, tuple) and len(condition) == 2:
                        operator, value = condition
                        if operator == '>':
                            filtered_recommendations = filtered_recommendations.loc[filtered_recommendations[key] > value]
                        elif operator == '<':
                            filtered_recommendations = filtered_recommendations.loc[filtered_recommendations[key] < value]
                        elif operator == '>=':
                            filtered_recommendations = filtered_recommendations.loc[filtered_recommendations[key] >= value]
                        elif operator == '<=':
                            filtered_recommendations = filtered_recommendations.loc[filtered_recommendations[key] <= value]
                        elif operator == '==':
                            filtered_recommendations = filtered_recommendations.loc[filtered_recommendations[key] == value]
                        else:
                            print(f"Unsupported operator: {operator}")
                    elif isinstance(condition, tuple) and len(condition) == 3 and condition[0] == 'between':
                        # Handle 'between' condition
                        lower_bound, upper_bound = condition[1], condition[2]
                        filtered_recommendations = filtered_recommendations.loc[
                            (filtered_recommendations[key] >= lower_bound) & (filtered_recommendations[key] <= upper_bound)
                        ]
                    else:
                        print(f"Invalid condition format: {condition}")
                else:
                    print(f"Warning: Column '{key}' not found in recommendations.")

            return filtered_recommendations
        else:
            return None