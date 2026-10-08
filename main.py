# Predictive Analytics for E-Commerce
# Sales Forecasting and Inventory Optimization Using Machine Learning
# Case Study: Sri Murugan Traders, Vellore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SALES_FILE = "Sales Data for Sep 2025 to Sep 2026..xlsx"
RETURN_FILE = "Sales Return Data Sep 2025 to Sep 2026..xlsx"

sales = pd.read_excel(SALES_FILE)
returns = pd.read_excel(RETURN_FILE)

print("Sales Shape:", sales.shape)
print("Return Shape:", returns.shape)
print("\nSales Columns:", sales.columns.tolist())
print("\nReturn Columns:", returns.columns.tolist())

# Date and numeric conversion
for col in ["Order Date", "Invoice Date", "Dispatch Date"]:
    if col in sales.columns:
        sales[col] = pd.to_datetime(sales[col], dayfirst=True, errors="coerce")

if "Returned Date" in returns.columns:
    returns["Returned Date"] = pd.to_datetime(
        returns["Returned Date"], dayfirst=True, errors="coerce"
    )

for col in ["Issued QTY", "Unit Price", "Base Price", "Total Amount + GST"]:
    if col in sales.columns:
        sales[col] = pd.to_numeric(sales[col], errors="coerce")

if "Returned Qty" in returns.columns:
    returns["Returned Qty"] = pd.to_numeric(
        returns["Returned Qty"], errors="coerce"
    )

sales = sales.drop_duplicates()
returns = returns.drop_duplicates()

sales = sales.dropna(subset=["Order Date", "Item Code", "Issued QTY"])
returns = returns.dropna(subset=["Returned Date", "Item Code", "Returned Qty"])

# Business summary
print("\n=== BUSINESS SUMMARY ===")
print("Total Sales Quantity:", sales["Issued QTY"].sum())
print("Total Revenue:", round(sales["Total Amount + GST"].sum(), 2))
print("Total Returned Quantity:", returns["Returned Qty"].sum())
print("Unique Products:", sales["Item Code"].nunique())
print("Unique Retailers:", sales["Retailer Code"].nunique())

# Daily sales and returns
daily_sales = sales.groupby("Order Date")["Issued QTY"].sum().reset_index()
daily_sales.columns = ["Date", "Sales"]

daily_returns = returns.groupby("Returned Date")["Returned Qty"].sum().reset_index()
daily_returns.columns = ["Date", "Returns"]

daily_data = pd.merge(daily_sales, daily_returns, on="Date", how="left")
daily_data["Returns"] = daily_data["Returns"].fillna(0)
daily_data["Net Sales"] = daily_data["Sales"] - daily_data["Returns"]
daily_data = daily_data.sort_values("Date")

# Monthly analysis
sales["Month"] = sales["Order Date"].dt.to_period("M")
monthly_sales = sales.groupby("Month")["Issued QTY"].sum().reset_index()
monthly_sales["Month"] = monthly_sales["Month"].astype(str)

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales["Month"], monthly_sales["Issued QTY"], marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Quantity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales_trend.png", dpi=200)
plt.show()

# Top products
top_products = (
    sales.groupby(["Item Code", "Item Name"])["Issued QTY"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n=== TOP 10 PRODUCTS ===")
print(top_products)

plt.figure(figsize=(12, 6))
top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Products by Sales Quantity")
plt.xlabel("Sales Quantity")
plt.tight_layout()
plt.savefig("top_products.png", dpi=200)
plt.show()

# Feature engineering
data = daily_data.copy()
data["Day"] = data["Date"].dt.day
data["Month"] = data["Date"].dt.month
data["Year"] = data["Date"].dt.year
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["Week"] = data["Date"].dt.isocalendar().week.astype(int)

data["Lag_1"] = data["Net Sales"].shift(1)
data["Lag_7"] = data["Net Sales"].shift(7)
data["Lag_14"] = data["Net Sales"].shift(14)
data["Lag_30"] = data["Net Sales"].shift(30)

data["Rolling_7"] = data["Net Sales"].rolling(7).mean()
data["Rolling_14"] = data["Net Sales"].rolling(14).mean()
data["Rolling_30"] = data["Net Sales"].rolling(30).mean()

data = data.dropna()

features = [
    "Day", "Month", "Year", "DayOfWeek", "Week",
    "Lag_1", "Lag_7", "Lag_14", "Lag_30",
    "Rolling_7", "Rolling_14", "Rolling_30"
]

X = data[features]
y = data["Net Sales"]

split = int(len(data) * 0.80)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
)
gb_model.fit(X_train, y_train)
gb_prediction = gb_model.predict(X_test)

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    non_zero = actual != 0
    mape = np.mean(
        np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])
    ) * 100
    return [name, mae, rmse, mape, r2]

results_df = pd.DataFrame(
    [
        evaluate_model("Random Forest", y_test, rf_prediction),
        evaluate_model("Gradient Boosting", y_test, gb_prediction),
    ],
    columns=["Model", "MAE", "RMSE", "MAPE (%)", "R2 Score"]
)

print("\n=== MODEL COMPARISON ===")
print(results_df)

best_model_name = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
best_model = rf_model if best_model_name == "Random Forest" else gb_model
print("\nBest Model:", best_model_name)

# Actual vs predicted
plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label="Actual")
plt.plot(rf_prediction, label="Random Forest")
plt.plot(gb_prediction, label="Gradient Boosting")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Days")
plt.ylabel("Net Sales")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=200)
plt.show()

# 30-day recursive forecast
future = data[["Date", "Net Sales"]].copy()
future_dates, future_predictions = [], []

for _ in range(30):
    next_date = future["Date"].max() + pd.Timedelta(days=1)

    row = pd.DataFrame({
        "Day": [next_date.day],
        "Month": [next_date.month],
        "Year": [next_date.year],
        "DayOfWeek": [next_date.dayofweek],
        "Week": [next_date.isocalendar().week],
        "Lag_1": [future["Net Sales"].iloc[-1]],
        "Lag_7": [future["Net Sales"].iloc[-7]],
        "Lag_14": [future["Net Sales"].iloc[-14]],
        "Lag_30": [future["Net Sales"].iloc[-30]],
        "Rolling_7": [future["Net Sales"].tail(7).mean()],
        "Rolling_14": [future["Net Sales"].tail(14).mean()],
        "Rolling_30": [future["Net Sales"].tail(30).mean()],
    })

    prediction = max(0, best_model.predict(row[features])[0])
    future_dates.append(next_date)
    future_predictions.append(prediction)

    future = pd.concat(
        [future, pd.DataFrame({"Date": [next_date], "Net Sales": [prediction]})],
        ignore_index=True
    )

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Sales": future_predictions
})

print("\n=== 30-DAY FORECAST ===")
print(forecast_df)

# Demand-based inventory calculation
average_daily_demand = data["Net Sales"].mean()
demand_std = data["Net Sales"].std()
LEAD_TIME_DAYS = 5
Z = 1.645

safety_stock = Z * demand_std * np.sqrt(LEAD_TIME_DAYS)
reorder_point = average_daily_demand * LEAD_TIME_DAYS + safety_stock

print("\n=== INVENTORY OPTIMIZATION ===")
print("Average Daily Demand:", round(average_daily_demand, 2))
print("Safety Stock:", round(safety_stock, 2))
print("Reorder Point:", round(reorder_point, 2))
print("Note: Actual stock levels require opening/closing stock and purchase data.")

# Product analysis
product_summary = (
    sales.groupby(["Item Code", "Item Name"])
    .agg(
        Total_Sales=("Issued QTY", "sum"),
        Average_Sales=("Issued QTY", "mean"),
        Number_of_Orders=("Invoice Number", "nunique"),
    )
    .reset_index()
    .sort_values("Total_Sales", ascending=False)
)

return_summary = (
    returns.groupby(["Item Code", "Item Name"])["Returned Qty"]
    .sum()
    .reset_index()
    .sort_values("Returned Qty", ascending=False)
)

# Export results
monthly_sales.to_excel("monthly_sales_analysis.xlsx", index=False)
product_summary.to_excel("product_sales_analysis.xlsx", index=False)
return_summary.to_excel("product_returns_analysis.xlsx", index=False)
forecast_df.to_excel("30_day_sales_forecast.xlsx", index=False)
results_df.to_excel("model_comparison.xlsx", index=False)

print("\n=== PROJECT COMPLETED ===")
print("Results exported successfully.")
