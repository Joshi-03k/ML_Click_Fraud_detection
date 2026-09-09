import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_and_prepare_data():

    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    # Get the folder where preprocessing.py is located
    folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    # Dataset path
    csv_path = os.path.join(
        folder,
        "click_fraud_dataset_cleaned.csv"
    )

    print("\nLooking for dataset:")
    print(csv_path)

    # Check dataset
    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            "\nERROR: Dataset not found!\n\n"
            "Expected file:\n"
            + csv_path +
            "\n\nMake sure the CSV is in the same "
            "folder as preprocessing.py."
        )

    # Load dataset
    df = pd.read_csv(csv_path)

    print("\nDataset loaded successfully!")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    # =====================================================
    # MISSING VALUES
    # =====================================================

    print("\nChecking missing values...")

    missing = df.isnull().sum().sum()

    print(
        "Total missing values:",
        missing
    )

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    print("\nCreating features...")

    # Timestamp features

    if "timestamp" in df.columns:

        timestamp = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df["hour"] = timestamp.dt.hour
        df["day"] = timestamp.dt.day
        df["month"] = timestamp.dt.month
        df["day_of_week"] = (
            timestamp.dt.dayofweek
        )

    # Referrer URL features

    if "referrer_url" in df.columns:

        referrer = (
            df["referrer_url"]
            .fillna("")
            .astype(str)
        )

        df["referrer_length"] = (
            referrer.str.len()
        )

        df["referrer_https"] = (
            referrer
            .str.startswith("https")
            .astype(int)
        )

    # Page URL features

    if "page_url" in df.columns:

        page = (
            df["page_url"]
            .fillna("")
            .astype(str)
        )

        df["page_url_length"] = (
            page.str.len()
        )

        df["page_https"] = (
            page
            .str.startswith("https")
            .astype(int)
        )

    # =====================================================
    # REMOVE IDENTIFIERS
    # =====================================================

    columns_to_remove = [
        "click_id",
        "user_id",
        "ip_address",
        "timestamp",
        "referrer_url",
        "page_url"
    ]

    existing_columns = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    df.drop(
        columns=existing_columns,
        inplace=True
    )

    print("\nRemoved columns:")
    print(existing_columns)

    # =====================================================
    # TARGET
    # =====================================================

    target = "is_fraudulent"

    if target not in df.columns:

        raise ValueError(
            "\nERROR: is_fraudulent column "
            "not found in dataset."
        )

    # Remove rows where target is missing

    df.dropna(
        subset=[target],
        inplace=True
    )

    # Features

    X = df.drop(
        columns=[target]
    )

    # Target

    y = pd.to_numeric(
        df[target]
    )

    # =====================================================
    # COLUMN IDENTIFICATION
    # =====================================================

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numeric_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # =====================================================
    # NUMERICAL PIPELINE
    # =====================================================

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # =====================================================
    # CATEGORICAL PIPELINE
    # =====================================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # =====================================================
    # PREPROCESSOR
    # =====================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    print(
        "\nPreprocessing pipeline created successfully!"
    )

    print("\nInput features:", X.shape[1])
    print("Target values:")
    print(y.value_counts())

    return X, y, preprocessor


# =========================================================
# RUN PREPROCESSING DIRECTLY
# =========================================================

if __name__ == "__main__":

    print("\nRunning preprocessing.py directly...")

    X, y, preprocessor = (
        load_and_prepare_data()
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nX shape:")
    print(X.shape)

    print("\ny shape:")
    print(y.shape)

    print("\nPreprocessor:")
    print(preprocessor)

    print("\nYou can now run your ML algorithm files.")