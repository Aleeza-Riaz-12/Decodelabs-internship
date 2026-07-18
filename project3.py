
# ============================================================
# Data Science Project 3
# Customer Segmentation using K-Means Clustering
# ============================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")


# Load Dataset
df = pd.read_excel("Dataset for Data Analytics(3).xlsx")

print("="*60)
print("First Five Rows")
print(df.head())

print("="*60)
print("Dataset Shape")
print(df.shape)

print("="*60)
print("Dataset Information")
print(df.info())

print("="*60)
print("Statistical Summary")
print(df.describe(include="all"))

# Check Missing Values
print("="*60)
print("Missing Values")
print(df.isnull().sum())

# Check Duplicate Records
print("="*60)
print("Duplicate Rows :", df.duplicated().sum())

# Remove Duplicate Records
df.drop_duplicates(inplace=True)

# Convert Date Column
df["Date"] = pd.to_datetime(df["Date"])

# Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"] .dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.day_name()

# Fill Missing Values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("="*60)
print("Missing Values After Cleaning")
print(df.isnull().sum())

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Distribution of Total Price
plt.figure(figsize=(8,5))
sns.histplot(df["TotalPrice"], bins=30, kde=True)
plt.title("Distribution of Total Price")
plt.show()

# Payment Method
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="PaymentMethod")
plt.xticks(rotation=45)
plt.title("Payment Method Distribution")
plt.show()

# Order Status
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="OrderStatus")
plt.xticks(rotation=45)
plt.title("Order Status Distribution")
plt.show()

# Quantity vs Total Price
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Quantity", y="TotalPrice")
plt.title("Quantity vs Total Price")
plt.show()
# ============================================================
# Encode Categorical Features
# ============================================================

label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("=" * 60)
print("Categorical Features Encoded Successfully")
print(df.head())


# ============================================================
# Feature Selection for Clustering
# ============================================================

X = df.drop(columns=["OrderID"])

print("=" * 60)
print("Selected Features Shape")
print(X.shape)


# ============================================================
# Feature Scaling
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("=" * 60)
print("Features Scaled Successfully")


# ============================================================
# Apply Principal Component Analysis (PCA)
# ============================================================

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

# Explained Variance Ratio
explained_variance = pca.explained_variance_ratio_

print("=" * 60)
print("Explained Variance Ratio")
print(explained_variance)

# Cumulative Explained Variance
cumulative_variance = np.cumsum(explained_variance)

print("=" * 60)
print("Cumulative Explained Variance")
print(cumulative_variance)


# ============================================================
# Plot Cumulative Explained Variance
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(range(1, len(cumulative_variance)+1),
         cumulative_variance,
         marker='o')

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)

plt.show()


# ============================================================
# Reduce Dataset into Two Principal Components
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca,
                      columns=["PC1","PC2"])

print("=" * 60)
print("PCA Dataset")
print(pca_df.head())


# ============================================================
# Visualize PCA Components
# ============================================================

plt.figure(figsize=(8,6))

plt.scatter(pca_df["PC1"],
            pca_df["PC2"])

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection of Dataset")

plt.show()


# ============================================================
# Elbow Method
# ============================================================

wcss = []

for i in range(1,11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_pca)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(1,11),
         wcss,
         marker='o')

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.grid(True)

plt.show()


# ============================================================
# Silhouette Score
# ============================================================

scores = []

for i in range(2,11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels)

    scores.append(score)

    print(f"Clusters = {i}  --->  Silhouette Score = {score:.4f}")

plt.figure(figsize=(8,5))

plt.plot(range(2,11),
         scores,
         marker="o")

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")

plt.grid(True)

plt.show()

# ============================================================
# Select Best Number of Clusters
# ============================================================

best_k = np.argmax(scores) + 2

print("=" * 60)
print("Best Number of Clusters :", best_k)


# ============================================================
# Build Final K-Means Model
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_pca)

pca_df["Cluster"] = clusters

print("=" * 60)
print("Cluster Labels Assigned Successfully")


# ============================================================
# Visualize Customer Clusters
# ============================================================

plt.figure(figsize=(10,7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set2",
    s=80
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color="black",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

plt.show()


# ============================================================
# Add Cluster Labels to Original Dataset
# ============================================================

df["Cluster"] = clusters

print("=" * 60)
print(df.head())


# ============================================================
# Number of Customers in Each Cluster
# ============================================================

print("=" * 60)
print("Customers in Each Cluster")

print(df["Cluster"].value_counts().sort_index())


# ============================================================
# Cluster-wise Statistics
# ============================================================

cluster_summary = df.groupby("Cluster").mean(numeric_only=True)

print("=" * 60)
print("Cluster Summary")

print(cluster_summary)


# ============================================================
# Cluster Size Visualization
# ============================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Cluster",
    palette="Set2"
)

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")

plt.show()


# ============================================================
# Average Total Price by Cluster
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Cluster",
    y="TotalPrice",
    estimator=np.mean,
    palette="Set2"
)

plt.title("Average Total Price by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Total Price")

plt.show()


# ============================================================
# Average Quantity by Cluster
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Cluster",
    y="Quantity",
    estimator=np.mean,
    palette="Set2"
)

plt.title("Average Quantity Purchased")
plt.xlabel("Cluster")
plt.ylabel("Average Quantity")

plt.show()


# ============================================================
# Customer Personas
# ============================================================

print("=" * 60)
print("Customer Personas")

for cluster in sorted(df["Cluster"].unique()):

    avg_price = df[df["Cluster"] == cluster]["TotalPrice"].mean()
    avg_quantity = df[df["Cluster"] == cluster]["Quantity"].mean()

    print("\nCluster", cluster)

    if avg_price >= df["TotalPrice"].mean():
        spending = "High Spending Customers"
    else:
        spending = "Budget Customers"

    if avg_quantity >= df["Quantity"].mean():
        purchase = "Bulk Buyers"
    else:
        purchase = "Occasional Buyers"

    print("Persona :", spending + " | " + purchase)

    print("Average Spending :", round(avg_price,2))
    print("Average Quantity :", round(avg_quantity,2))


# ============================================================
# Business Insights
# ============================================================

print("=" * 60)
print("Business Recommendations")

print("""
1. High Spending Customers should receive premium offers and loyalty rewards.

2. Budget Customers can be targeted using discounts and promotional campaigns.

3. Bulk Buyers should receive bundle offers and quantity-based discounts.

4. Occasional Buyers can be encouraged through personalized marketing.

5. Customer segmentation helps businesses improve marketing efficiency,
   increase sales, and enhance customer satisfaction.
""")


# ============================================================
# Conclusion
# ============================================================

print("=" * 60)
print("PROJECT CONCLUSION")

print("""
Principal Component Analysis (PCA) successfully reduced the dimensionality
of the dataset while preserving most of the important information.

The Elbow Method and Silhouette Score were used to determine the optimal
number of clusters for K-Means clustering.

K-Means successfully grouped customers into meaningful segments based on
their purchasing behavior.

These customer segments can help businesses design targeted marketing
campaigns, improve customer retention, optimize product recommendations,
and make better business decisions through data-driven insights.
""");