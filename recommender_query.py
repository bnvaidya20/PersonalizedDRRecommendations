import config

from utils.utils import DataPreprocessor
from utils.recommender_scheme import RecommendationQuery


# Instantiate  Data with Recommendation
recommendation_query = RecommendationQuery(config.file_path_reco_test)
# recommendation_query = RecommendationQuery(config.file_path_reco_entire)

# Load recommendations
recommendation_query.load_recommendations()

# Query recommendations:
user_query = recommendation_query.get_recommendation(hour=('between', 17, 20), is_summer= ('==', 1), is_weekend=('==', 0))

preprocessor = DataPreprocessor(user_query)

preprocessor.drop_columns(config.columns_name_to_drop)
user_query_final = preprocessor.get_preprocessed_data()

print(user_query_final.head())
print(user_query_final.tail())

