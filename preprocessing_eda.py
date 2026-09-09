import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# FIND CSV FILE
# =====================================================

folder = os.path.dirname(
    os.path.abspath(__file__)
)

csv_path = os.path.join(
    folder,
    "click_fraud_dataset_cleaned.csv"
)

# Check file
if not os.path.exists(csv_path):

    print("\nERROR")
    print("Dataset not found.")

    print("\nPython searched here:")
    print(csv_path)

    print(
        "\nPut click_fraud_dataset_cleaned.csv "
        "in the same folder as preprocessing_eda.py"
    )

    exit()

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading dataset...")

df = pd.read_csv(csv_path)

print("Dataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# TARGET DISTRIBUTION
# =====================================================

print("\nTarget Distribution:")

print(
    df["is_fraudulent"].value_counts()
)

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="is_fraudulent"
)

plt.title(
    "Fraudulent vs Non-Fraudulent Clicks"
)

plt.show()

# =====================================================
# BOXPLOT
# =====================================================

if "click_duration" in df.columns:

    plt.figure(figsize=(8,5))

    sns.boxplot(
        data=df,
        x="is_fraudulent",
        y="click_duration"
    )

    plt.title(
        "Click Duration vs Fraud"
    )

    plt.show()

# =====================================================
# SCATTER PLOT
# =====================================================

if (
    "click_duration" in df.columns and
    "bot_likelihood_score" in df.columns
):

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        data=df,
        x="click_duration",
        y="bot_likelihood_score",
        hue="is_fraudulent",
        alpha=0.5
    )

    plt.title(
        "Click Duration vs Bot Score"
    )

    plt.show()

# =====================================================
# HEATMAP
# =====================================================

numeric_df = df.select_dtypes(
    include=["int64","float64"]
)

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.show()

print("\nEDA Completed Successfully.")