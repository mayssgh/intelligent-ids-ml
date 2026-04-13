import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """Load a sample of dataset for faster processing."""
    
    # Load only first 100,000 rows (fast & enough for training)
    df = pd.read_csv(file_path, nrows=100000)
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix column names
    df.columns = df.columns.str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill missing values
    df = df.ffill()

    # Drop remaining NaN if any
    df = df.dropna()

    return df


def encode_features(df: pd.DataFrame):
    """Encode categorical columns."""
    
    df_encoded = df.copy()
    encoders = {}

    for col in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    return df_encoded, encoders


def scale_features(X: pd.DataFrame):
    """Scale numerical features."""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split dataset into train and test sets."""
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preprocess_pipeline(file_path: str, target_column: str):
    """
    Full preprocessing pipeline.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, encoders
    """

    # Load
    df = load_data(file_path)

    # Clean
    df = clean_data(df)

    # Separate target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Encode categorical features
    X_encoded, encoders = encode_features(X)

    # Scale features
    X_scaled, scaler = scale_features(X_encoded)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    return X_train, X_test, y_train, y_test, scaler, encoders


# Run as script
if __name__ == "__main__":
    FILE_PATH = "data/raw/Wednesday-workingHours.pcap_ISCX.csv"
    TARGET = "Label"

    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_pipeline(
        FILE_PATH, TARGET
    )

    print("✅ Preprocessing completed.")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")