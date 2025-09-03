# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -----------------------------
# Task 1: Load and Explore Dataset
# -----------------------------

# Load Iris dataset from sklearn
iris = load_iris()

# Convert to pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Explore structure
print("\nData types and missing values:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# No missing values in Iris dataset, but if there were, we could fill/drop
# Example: df.fillna(df.mean(), inplace=True)

# -----------------------------
# Task 2: Basic Data Analysis
# -----------------------------

# Basic statistics
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Grouping by species and calculating mean
grouped = df.groupby('species').mean()
print("\nMean values per species:")
print(grouped)

# Example observation
print("\nObservation: Versicolor has intermediate measurements between Setosa and Virginica.")

# -----------------------------
# Task 3: Data Visualization
# -----------------------------

# Set seaborn style for better plots
sns.set(style="whitegrid")

# 1️⃣ Line chart: We'll simulate a trend using cumulative mean of sepal length
plt.figure(figsize=(8, 5))
df['sepal length (cm)'].cumsum().plot()
plt.title("Cumulative Sepal Length Trend")
plt.xlabel("Sample Index")
plt.ylabel("Cumulative Sepal Length (cm)")
plt.show()

# 2️⃣ Bar chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3️⃣ Histogram: Distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4️⃣ Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='Set1')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
