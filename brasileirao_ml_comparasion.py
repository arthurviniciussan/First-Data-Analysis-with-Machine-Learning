import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

csv_path = "Brasileirao_Matches.csv"

df = pd.read_csv(csv_path)
df = df.dropna(subset=["home_goal", "away_goal"]).copy()
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df["match_id"] = df.index

home_rows = df[
    ["match_id", "datetime", "home_team", "home_goal", "away_goal"]
].copy()
home_rows.columns = ["match_id", "datetime", "team", "gf", "ga"]

away_rows = df[
    ["match_id", "datetime", "away_team", "away_goal", "home_goal"]
].copy()
away_rows.columns = ["match_id", "datetime", "team", "gf", "ga"]

team_matches = pd.concat([home_rows, away_rows]).sort_values(
    ["team", "datetime"]
)

team_matches["gf_avg_before"] = (
    team_matches.groupby("team")["gf"]
    .transform(lambda series: series.shift().expanding().mean())
)

team_matches["ga_avg_before"] = (
    team_matches.groupby("team")["ga"]
    .transform(lambda series: series.shift().expanding().mean())
)

home_features = team_matches.rename(
    columns={
        "gf_avg_before": "home_gf_avg",
        "ga_avg_before": "home_ga_avg",
    }
)[["match_id", "team", "home_gf_avg", "home_ga_avg"]]

df = df.merge(
    home_features,
    left_on=["match_id", "home_team"],
    right_on=["match_id", "team"],
    how="left",
).drop(columns="team")

away_features = team_matches.rename(
    columns={
        "gf_avg_before": "away_gf_avg",
        "ga_avg_before": "away_ga_avg",
    }
)[["match_id", "team", "away_gf_avg", "away_ga_avg"]]

df = df.merge(
    away_features,
    left_on=["match_id", "away_team"],
    right_on=["match_id", "team"],
    how="left",
).drop(columns="team")

df["goal_diff"] = df["home_goal"] - df["away_goal"]

features = [
    "home_gf_avg",
    "home_ga_avg",
    "away_gf_avg",
    "away_ga_avg",
]
target = "goal_diff"

model_df = df.dropna(subset=features).copy()

X = model_df[features]
y = model_df[target]

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

df_model = df.dropna(subset=features).copy()

season = 2022
home_team = "Flamengo-RJ"
away_team = "Fluminense-RJ"

match = df_model[
    (df_model["season"] == season)
    & (df_model["home_team"] == home_team)
    & (df_model["away_team"] == away_team)
].iloc[0]

input_data = pd.DataFrame([match[features]])

actual_goal_diff = match[target]
actual_home_goals = match["home_goal"]
actual_away_goals = match["away_goal"]

linear_prediction = linear_regression.predict(input_data)[0]
forest_prediction = random_forest.predict(input_data)[0]

print(f"Match: {home_team} vs {away_team} ({season})")
print(f"Actual score: {actual_home_goals:.0f} - {actual_away_goals:.0f}")
print(f"Actual goal difference: {actual_goal_diff:+.0f}")
print(f"Linear Regression prediction: {linear_prediction:+.2f}")
print(f"Random Forest prediction: {forest_prediction:+.2f}")


import matplotlib.pyplot as plt

methods = results["Method"]
test_r2 = results["Test R2"]

plt.figure(figsize=(7, 5))
plt.bar(methods, test_r2)

plt.ylabel("Test R²")
plt.title("Brazilian Championship: Model Comparison")

plt.tight_layout()
plt.show()