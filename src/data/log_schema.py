import pandas as pd
import json
import datetime

data = pd.read_csv("data/raw/dataset.csv")

schema = {
    'source': "Predict students' dropout and academic success from Kaggle",
    'url': "https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention?select=dataset.csv",
    "data_collected": str(datetime.date.today()),
    "columns": {col : "Categorical" if data[col].dtype == 'object' else "Numerical" for col in data.columns},
    "num_rows": data.shape[0],
    "num_columns": data.shape[1]
}

print(json.dumps(schema, indent=4))
with open("data/raw/log_schema.json", "w") as f:
    json.dump(schema, f, indent=4)

print("Schema logged to data/raw/log_schema.json")# Script to log the schema of the dataset