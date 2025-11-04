import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample

# Load dataset
data_path = r"D:\prajwal\Documents\inventory_demand_forecast\data.csv\data.csv"

df = pd.read_csv(data_path)

# Drop rows with missing values
df.dropna(inplace=True)

# Add price_per_unit feature
df["price_per_unit"] = df["total_price"] / (df["units_sold"] + 1)  # prevent div by 0

# Target and features
target = "units_sold"
features = ['store_id', 'sku_id', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku', 'price_per_unit']

# Prepare DataFrame
df = df[features + [target]]

# Split into high and low demand (threshold = 20 for better balancing)
high_demand = df[df[target] > 20]
low_demand = df[df[target] <= 20]

# Upsample high demand 3x of low demand
high_demand_upsampled = resample(
    high_demand,
    replace=True,
    n_samples=len(low_demand) * 3,
    random_state=42
)

# Combine and shuffle
df_balanced = pd.concat([low_demand, high_demand_upsampled]).sample(frac=1, random_state=42)

# Features and target
X = df_balanced[features]
y = df_balanced[target]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) * 100

print("\n📊 Model Accuracy:")
print(f"MSE:  {mse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.2f}%")

# Test with custom high-value input (force high features)
sample_input = pd.DataFrame([{
    "store_id": X["store_id"].max(),
    "sku_id": X["sku_id"].max(),
    "total_price": 1000,
    "base_price": 200,
    "is_featured_sku": 1,
    "is_display_sku": 1,
    "price_per_unit": 50
}])
sample_scaled = scaler.transform(sample_input)
sample_prediction = model.predict(sample_scaled)[0]
print(f"\n🧪 Sample High-Demand Prediction: {sample_prediction:.2f} units")

# Save everything
joblib.dump(model, "backend/model.pkl")
joblib.dump(scaler, "backend/scaler.pkl")
joblib.dump(df[target].max(), "backend/max_demand.pkl")
# Save R² score to file so frontend can read it
joblib.dump(r2, "backend/r2_score.pkl")
print(f"✅ R² score ({r2:.2f}%) saved successfully!")

valid_ids = {
    "store_ids": df["store_id"].unique().tolist(),
    "sku_ids": df["sku_id"].unique().tolist()
}
joblib.dump(valid_ids, "backend/valid_ids.pkl")

print("\n✅ Model training complete. Model and support files saved.")


