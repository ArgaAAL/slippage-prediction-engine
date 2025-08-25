import joblib
import json

# Load the list from the pickle file
feature_list = joblib.load('models/feature_columns.pkl')

# Print it in a clean format
print(json.dumps(feature_list, indent=2))