#%%
# Import features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
import os
import json

#%%
# Load datasets and merge listings with calendar

listings_path = r"D:\Rami\Personal\Airbnb_Dynamic_Pricing\Data\listings.csv.gz"
calendar_path = r"D:\Rami\Personal\Airbnb_Dynamic_Pricing\Data\calendar.csv.gz"

base_path = r"D:\Rami\Personal\Airbnb_Dynamic_Pricing"

center_lat = 52.3676
center_lon = 4.9041

listings = pd.read_csv(listings_path)
calendar = pd.read_csv(calendar_path)

listings = listings[[
"id", "price",

"latitude", "longitude",
"neighbourhood_cleansed",

"property_type", "room_type",
"accommodates", "bathrooms", "bedrooms", "beds",

"minimum_nights", "maximum_nights",

"number_of_reviews",
"number_of_reviews_ltm",
"number_of_reviews_l30d",
"reviews_per_month",

"review_scores_rating",
"review_scores_accuracy",
"review_scores_cleanliness",
"review_scores_checkin",
"review_scores_communication",
"review_scores_location",
"review_scores_value",

"host_is_superhost",
"host_response_rate",
"host_acceptance_rate",
"instant_bookable",

"host_listings_count",
"host_total_listings_count",
"calculated_host_listings_count",
"calculated_host_listings_count_entire_homes",
"calculated_host_listings_count_private_rooms",
"calculated_host_listings_count_shared_rooms"
]]

data = listings.merge(calendar, left_on='id', right_on='listing_id',validate="one_to_many")

# %%
# Create target variable for booking

booked = data["available"].map({"t": 0, "f": 1})
data["booked"] = booked
data = data.drop(['price_y', 'minimum_nights_y', 'maximum_nights_y', 'available', 'adjusted_price'], axis = 1)

# %%
# Define distance computation function

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# %%
# Cleaning data

date = pd.to_datetime(data["date"])
data["month"] = date.dt.month

data["price_x"] = data["price_x"].str.replace(r"[$,]", "", regex=True).astype(float)
data = data.dropna(subset=["price_x"])

data["host_is_superhost"] = data["host_is_superhost"].map({"t": 1, "f": 0})
data["instant_bookable"] = data["instant_bookable"].map({"t": 1, "f": 0})

data["host_response_rate"] = data["host_response_rate"].str.replace("%", "").astype(float) / 100
data["host_acceptance_rate"] = data["host_acceptance_rate"].str.replace("%", "").astype(float) / 100

data["distance_to_center"] = haversine(
    data["latitude"], data["longitude"],
    center_lat, center_lon
)

num_cols = data.select_dtypes(include=["float64", "int64"]).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

cat_cols = data.select_dtypes(include=["object"]).columns
data[cat_cols] = data[cat_cols].fillna("unknown")

# %%
# Split data into train and test sets (80% training, 20% testing)

unique_ids = data["id"].unique()

train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

train_data = data[data["id"].isin(train_ids)]
test_data  = data[data["id"].isin(test_ids)]

neigh_avg_price = train_data.groupby("neighbourhood_cleansed")["price_x"].mean()

train_data["neigh_avg_price"] = train_data["neighbourhood_cleansed"].map(neigh_avg_price)
test_data["neigh_avg_price"] = test_data["neighbourhood_cleansed"].map(neigh_avg_price)

global_price_mean = train_data["price_x"].mean()
test_data["neigh_avg_price"] = test_data["neigh_avg_price"].fillna(global_price_mean)

train_data["relative_price"] = train_data["price_x"] / train_data["neigh_avg_price"]
test_data["relative_price"] = test_data["price_x"] / test_data["neigh_avg_price"]

train_data["price_diff"] = train_data["price_x"] - train_data["neigh_avg_price"]
test_data["price_diff"] = test_data["price_x"] - test_data["neigh_avg_price"]

train_data["host_quality_score"] = (
    train_data["host_is_superhost"].fillna(0)
    + train_data["host_response_rate"].fillna(0)
    + train_data["host_acceptance_rate"].fillna(0)
)

test_data["host_quality_score"] = (
    test_data["host_is_superhost"].fillna(0)
    + test_data["host_response_rate"].fillna(0)
    + test_data["host_acceptance_rate"].fillna(0)
)

train_data["review_activity"] = train_data["number_of_reviews_l30d"] / (train_data["number_of_reviews"] + 1)
test_data["review_activity"] = test_data["number_of_reviews_l30d"] / (test_data["number_of_reviews"] + 1)

train_data["high_season"] = train_data["month"].isin([6, 7, 8]).astype(int)
test_data["high_season"] = test_data["month"].isin([6, 7, 8]).astype(int)

train_data = pd.get_dummies(train_data, columns=[
    "property_type",
    "room_type",
    "neighbourhood_cleansed"
], drop_first=True)

test_data = pd.get_dummies(test_data, columns=[
    "property_type",
    "room_type",
    "neighbourhood_cleansed"
], drop_first=True)

train_data, test_data = train_data.align(test_data, join="left", axis=1, fill_value=0)

X_train = train_data.drop(columns=["booked", "date", "id", "listing_id"])
y_train = train_data["booked"]

X_test = test_data.drop(columns=["booked", "date", "id", "listing_id"])
y_test = test_data["booked"]

# %%
# Train XGBoost model with class balancing

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print("Negative (0):", neg)
print("Positive (1):", pos)
print("scale_pos_weight:", scale_pos_weight)

model = XGBClassifier(n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
# Evaluate model performance

y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc)
}

with open(os.path.join(base_path, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# %%
# Prepare feature names and compute SHAP values

feature_names_clean = {
    "price_x": "Price (€)",
    "distance_to_center": "Distance to City Center (km)",
    "maximum_nights_x": "Max Nights",
    "minimum_nights_x": "Min Nights",
    "host_acceptance_rate": "Host Acceptance Rate",
    "host_quality_score": "Host Quality",
    "reviews_per_month": "Reviews per Month",
    "number_of_reviews": "Total Reviews",
    "number_of_reviews_ltm": "Reviews (Last 12 Months)",
    "property_type_Entire townhouse": "Entire Townhouse",
    "relative_price": "Relative Price",
    "price_diff": "Price vs Area Avg",
    "room_type_Private room": "Private Room",
    "month": "Month",
    "bedrooms": "Bedrooms",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "review_scores_value": "Value Rating",
    "host_listings_count": "Host Listings",
    "calculated_host_listings_count": "Host Portfolio Size",
    "accommodates": "Guest Capacity",
    "review_scores_rating": "Overall Rating",
    "review_scores_cleanliness": "Cleanliness Rating",
    "review_scores_checkin": "Check-in Rating",
    "review_scores_communication": "Communication Rating",
    "review_scores_location": "Location Rating",
    "host_response_rate": "Host Response Rate",
    "host_is_superhost": "Superhost Status",
    "instant_bookable": "Instant Booking"
}

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

X_test_named = X_test.rename(columns=feature_names_clean)

shap.summary_plot(shap_values, X_test_named, show=False)
plt.title("Global Drivers of Booking Probability")
plt.savefig(os.path.join(base_path, "Shap_Summary.png"), dpi=300, bbox_inches='tight')


# %%
# Computing confusion matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Booking rate (y=1):", y_test.mean())

# %%
# Checking feature importance

shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": shap_importance
}).sort_values(by="importance", ascending=False)

print(shap_df.head(15))

for feature in shap_df["feature"].head(10):
    corr = np.corrcoef(X_test[feature], shap_values[:, X_test.columns.get_loc(feature)])[0,1]
    print(feature, "correlation with SHAP:", round(corr, 3))

# %%
# Sample listings and rank by predicted probability

sample_idx = test_data.drop_duplicates("id").sample(20, random_state=42).index

X_sample = X_test.loc[sample_idx]
meta_sample = data.loc[sample_idx]

probs = model.predict_proba(X_sample)[:, 1]

results = pd.DataFrame({
    "id": meta_sample["id"],
    "price": meta_sample["price_x"],
    "neighbourhood": meta_sample["neighbourhood_cleansed"],
    "room_type": meta_sample["room_type"],
    "bedrooms": meta_sample["bedrooms"],
    "probability": probs
})

results = results.sort_values(by="probability", ascending=False)

print(results)

# %%
# Compute shap values

top_idx = results.index[0]

top_row = X_test.loc[[top_idx]]

top_shap = shap_values[X_test.index.get_loc(top_idx)]

shap_explain = pd.DataFrame({
    "feature": X_test.columns,
    "shap_value": top_shap
}).sort_values(by="shap_value", ascending=False)

print(shap_explain.head(10))  
print(shap_explain.tail(10))  

#%%
# Apply price scenarios to all the test listings

X_temp = X_test.copy()
meta_temp = data.loc[X_temp.index]

price_range = np.linspace(0.80, 1.20, 21)  # -20% to +20%

results_price_change = []

for factor in price_range:
    
    X_mod = X_temp.copy()
    X_mod["price_x"] = X_mod["price_x"] * factor
    
    probs = model.predict_proba(X_mod)[:, 1]
    
    df = pd.DataFrame({
        "id": meta_temp["id"],
        "price_factor": factor,
        "price": X_mod["price_x"],
        "probability": probs
    })
    
    results_price_change.append(df)

results_price_change = pd.concat(results_price_change, ignore_index=True)

print(results_price_change.head())
print(results_price_change.groupby("price_factor")["probability"].mean())


#%% 
# Explain the top ranked listing

shap_df_plot = shap_df.copy()
shap_df_plot["feature"] = shap_df_plot["feature"].map(feature_names_clean).fillna(shap_df_plot["feature"])


top_shap_plot = shap_explain.copy()
top_shap_plot["feature"] = top_shap_plot["feature"].map(feature_names_clean).fillna(top_shap_plot["feature"])

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

# Plot feature importance

plt.figure(figsize=(8, 5))

top_features = shap_df_plot.head(10)

sns.barplot(
    x=top_features["importance"],
    y=top_features["feature"],
    palette="viridis"
)

plt.xlabel("Impact on Booking Probability")
plt.ylabel("Feature")
plt.title("Key Drivers of Booking Likelihood")
plt.tight_layout()

plt.savefig(os.path.join(base_path, "Feature_Importance.png"), dpi=300, bbox_inches='tight')

# Plot price vs booking probability

q99 = X_test["price_x"].quantile(0.99)
X_plot = X_test[X_test["price_x"] <= q99]

df_plot = pd.DataFrame({
    "price": X_plot["price_x"],
    "prob": model.predict_proba(X_plot)[:, 1]
})

smoothed = lowess(df_plot["prob"], df_plot["price"], frac=0.2)

plt.figure(figsize=(8, 5))

hb = plt.hexbin(
    df_plot["price"],
    df_plot["prob"],
    gridsize=50,
    cmap="viridis"
)

plt.plot(
    smoothed[:, 0],
    smoothed[:, 1],
    color="#FF4B4B",
    linewidth=3,
    label="Trend"
)

plt.xlabel("Price (€)")
plt.ylabel("Booking Probability")
plt.title("Booking Probability vs Price")

cb = plt.colorbar(hb)
cb.set_label("Density")

plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(base_path, "Price_vs_Probability.png"), dpi=300, bbox_inches='tight')

# Plot price sensitivity distribution

pivot = results_price_change.pivot_table(
    index="id",
    columns="price_factor",
    values="probability"
)

base = 1.0
low = 0.8
high = 1.2

sens_up = (pivot[high] - pivot[base]).dropna()    
sens_down = (pivot[low] - pivot[base]).dropna()   

xmin = min(sens_up.min(), sens_down.min())
xmax = max(sens_up.max(), sens_down.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

sns.histplot(
    sens_up,
    bins=50,
    kde=True,
    color="#4C72B0",
    ax=axes[0]
)

axes[0].set_xlim(xmin, xmax)
axes[0].set_title("+20% Price Increase")
axes[0].set_xlabel("Change in Booking Probability")
axes[0].set_ylabel("Number of Listings")

sns.histplot(
    sens_down,
    bins=50,
    kde=True,
    color="#55A868",
    ax=axes[1]
)


axes[1].set_xlim(xmin, xmax)
axes[1].set_title("-20% Price Decrease")
axes[1].set_xlabel("Change in Booking Probability")

plt.suptitle("Asymmetric Price Sensitivity (±20%)")

plt.tight_layout()

plt.savefig(
    os.path.join(base_path, "Price_Sensitivity.png"),
    dpi=300,
    bbox_inches='tight'
)


# Plot location effect

sample = X_plot.sample(5000, random_state=42)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=sample["distance_to_center"],
    y=model.predict_proba(sample)[:, 1],
    alpha=0.2,
    color="#4C72B0"
)

smooth_loc = lowess(
    model.predict_proba(sample)[:, 1],
    sample["distance_to_center"],
    frac=0.2
)

plt.plot(
    smooth_loc[:, 0],
    smooth_loc[:, 1],
    color="#FF4B4B",
    linewidth=3
)

plt.xlabel("Distance to City Center (km)")
plt.ylabel("Booking Probability")
plt.title("Effect of Location on Booking Probability")

plt.tight_layout()

plt.savefig(os.path.join(base_path, "Location_Effect.png"), dpi=300, bbox_inches='tight')

# Probability against price change

df_plot = results_price_change.copy()
df_plot["price_change_pct"] = (df_plot["price_factor"] - 1.0) * 100
line_df = df_plot.groupby("price_change_pct")["probability"].mean().reset_index()
line_df = line_df.sort_values("price_change_pct")
plt.figure(figsize=(8, 5))

plt.plot(
    line_df["price_change_pct"],
    line_df["probability"],
    color="#4C72B0",
    linewidth=3
)

p0 = line_df.loc[line_df["price_change_pct"] == 0, "probability"].values[0]
plt.axhline(p0, color="#FF4B4B", linestyle="--", linewidth=1)
plt.axvline(0, color="#FF4B4B", linestyle="--", linewidth=1)

plt.xlabel("Price Change (%)")
plt.ylabel("Booking Probability")
plt.title("Booking Probability vs Price Change")

plt.tight_layout()

plt.savefig(
    os.path.join(base_path, "Price_Change_vs_Probability.png"),
    dpi=300,
    bbox_inches='tight'
)

# Plot SHAP explanation for one listing

plt.figure(figsize=(8, 5))

top_shap_plot = top_shap_plot.head(10)

sns.barplot(
    x=top_shap_plot["shap_value"],
    y=top_shap_plot["feature"],
    palette="magma"
)

plt.xlabel("Contribution to Booking Probability")
plt.ylabel("Feature")
plt.title("Drivers of a High-Performing Listing")

plt.tight_layout()

plt.savefig(os.path.join(base_path, "Top_Listing_Shap.png"), dpi=300, bbox_inches='tight')

