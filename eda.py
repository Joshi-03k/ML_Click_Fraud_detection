import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# LOAD DATA
# =====================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

folder = os.path.dirname(
    os.path.abspath(__file__)
)

csv_path = os.path.join(
    folder,
    "click_fraud_dataset_cleaned.csv"
)

if not os.path.exists(csv_path):

    raise FileNotFoundError(
        "\nDataset not found:\n"
        + csv_path
    )

df = pd.read_csv(csv_path)

print("\nDataset loaded successfully!")

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\nDataset Shape:")
print(df.shape)

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe(include="all"))

# =====================================================
# MISSING VALUES
# =====================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print(
    "\nTotal Missing Values:",
    df.isnull().sum().sum()
)

# =====================================================
# DUPLICATES
# =====================================================

print(
    "\nDuplicate Rows:",
    df.duplicated().sum()
)

# =====================================================
# TARGET DISTRIBUTION
# =====================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(
    df["is_fraudulent"].value_counts()
)

print("\nTarget Percentage:")

print(
    df["is_fraudulent"]
    .value_counts(normalize=True) * 100
)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="is_fraudulent"
)

plt.title(
    "Fraudulent vs Non-Fraudulent Clicks"
)

plt.xlabel("Is Fraudulent")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# =====================================================
# NUMERICAL COLUMNS
# =====================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical Columns:")
print(numeric_columns)

# =====================================================
# CATEGORICAL COLUMNS
# =====================================================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

# =====================================================
# BOXPLOTS
# =====================================================

box_columns = [

    "click_duration",

    "scroll_depth",

    "mouse_movement",

    "keystrokes_detected",

    "click_frequency",

    "time_since_last_click",

    "bot_likelihood_score"
]

print("\nGenerating Boxplots...")

for column in box_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="is_fraudulent",
        y=column
    )

    plt.title(
        column +
        " vs Fraudulent Status"
    )

    plt.xlabel("Is Fraudulent")
    plt.ylabel(column)

    plt.tight_layout()
    plt.show()

# =====================================================
# SCATTER PLOT 1
# =====================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="click_duration",
    y="bot_likelihood_score",
    hue="is_fraudulent",
    alpha=0.5
)

plt.title(
    "Click Duration vs Bot Likelihood Score"
)

plt.xlabel("Click Duration")
plt.ylabel("Bot Likelihood Score")

plt.tight_layout()
plt.show()

# =====================================================
# SCATTER PLOT 2
# =====================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="click_frequency",
    y="time_since_last_click",
    hue="is_fraudulent",
    alpha=0.5
)

plt.title(
    "Click Frequency vs Time Since Last Click"
)

plt.xlabel("Click Frequency")
plt.ylabel("Time Since Last Click")

plt.tight_layout()
plt.show()

# =====================================================
# SCATTER PLOT 3
# =====================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="mouse_movement",
    y="scroll_depth",
    hue="is_fraudulent",
    alpha=0.5
)

plt.title(
    "Mouse Movement vs Scroll Depth"
)

plt.xlabel("Mouse Movement")
plt.ylabel("Scroll Depth")

plt.tight_layout()
plt.show()

# =====================================================
# CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(12, 8))

correlation = df[
    numeric_columns
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()
plt.show()

print(
    "\nEDA Completed Successfully."
)