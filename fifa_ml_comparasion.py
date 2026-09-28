import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("fifa_ranking.csv")

features = [
    "rank_change",
    "cur_year_avg",
    "last_year_avg",
    "two_year_ago_avg",
    "three_year_ago_avg",
]
target = "total_points"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

linear_train_pred = linear_regression.predict(X_train)
linear_test_pred = linear_regression.predict(X_test)

linear_results = {
    "Method": "Linear Regression",
    "Training MSE": mean_squared_error(y_train, linear_train_pred),
    "Training R2": r2_score(y_train, linear_train_pred),
    "Test MSE": mean_squared_error(y_test, linear_test_pred),
    "Test R2": r2_score(y_test, linear_test_pred),
}

random_forest = RandomForestRegressor(max_depth=2, random_state=100)
random_forest.fit(X_train, y_train)

forest_train_pred = random_forest.predict(X_train)
forest_test_pred = random_forest.predict(X_test)

forest_results = {
    "Method": "Random Forest",
    "Training MSE": mean_squared_error(y_train, forest_train_pred),
    "Training R2": r2_score(y_train, forest_train_pred),
    "Test MSE": mean_squared_error(y_test, forest_test_pred),
    "Test R2": r2_score(y_test, forest_test_pred),
}

results = pd.DataFrame([linear_results, forest_results])

print(results.to_string(index=False))

import matplotlib.pyplot as plt

methods = results["Method"]
test_r2 = results["Test R2"]

plt.figure(figsize=(7, 5))
plt.bar(methods, test_r2)

plt.ylabel("Test R²")
plt.title("FIFA Ranking: Model Comparison")
plt.ylim(0, 1.05)

plt.tight_layout()
plt.show()