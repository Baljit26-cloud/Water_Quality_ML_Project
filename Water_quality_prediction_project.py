import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Data Loading
df = pd.read_csv("C:\\Users\\bk572\\Downloads\\water_potability.csv")

# Data Cleaning
water_data = df.fillna({
    'ph': df['ph'].mean(),
    'Sulfate': df['Sulfate'].mean(),
    'Trihalomethanes': df['Trihalomethanes'].mean()
})

#
def remove_outliers_iqr(df, col, lower_percentile=0.01, upper_percentile=0.99):
    Q1 = df[col].quantile(lower_percentile)
    Q3 = df[col].quantile(upper_percentile)
    
    df_cleaned = df[(df[col] >= Q1) & (df[col] <= Q3)]
    return df_cleaned

initial_rows = len(water_data)

water_data = remove_outliers_iqr(water_data, 'ph')
water_data = remove_outliers_iqr(water_data, 'Sulfate')

top_5_features = ['Sulfate', 'ph', 'Hardness', 'Chloramines', 'Solids']
final_columns = top_5_features + ['Potability'] 
water_data = water_data[final_columns]

X = water_data.drop('Potability', axis='columns')
y = water_data['Potability']
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(
    n_estimators=1000, 
    class_weight = "balanced",
    
    
)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {final_accuracy * 100:.2f}%")
importances = model.feature_importances_
feature_names = X.columns
feature_ranking = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Score': importances
})
feature_ranking = feature_ranking.sort_values(by='Importance_Score', ascending=False)
print(feature_ranking)


import joblib


joblib.dump(model, 'random_forest_model.pkl')


feature_names = X.columns.tolist()
joblib.dump(feature_names, 'model_features.pkl')



