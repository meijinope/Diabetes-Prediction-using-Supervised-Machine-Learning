import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import glob, os

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# Dataset Analysis
# ---------------------------
def analyze_dataset():
    """Perform quick EDA on the diabetes dataset"""
    files = glob.glob("datasets/diabetes*")
    if not files:
        raise FileNotFoundError("Could not find diabetes dataset in 'datasets/'")
    df = pd.read_csv(files[0])

    print("=== DATASET ANALYSIS ===")
    print(f"Shape: {df.shape}")
    print(f"\nFeatures: {list(df.columns[:-1])}")

    print("\nTarget distribution:")
    print(df['Diagnosis'].value_counts())

    print("\nTarget percentages:")
    print((df['Diagnosis'].value_counts(normalize=True) * 100).round(1))

    print("\nMissing values:")
    print(df.isnull().sum()[lambda x: x > 0] if df.isnull().sum().sum() else "None ✓")

    print("\nFeature statistics:")
    print(df.describe())

    return df


# ---------------------------
# Data Preprocessing
# ---------------------------
def load_and_preprocess_data():
    """Load, clean, and preprocess the diabetes dataset"""
    files = glob.glob("datasets/diabetes*")
    if not files:
        raise FileNotFoundError("Could not find diabetes dataset in 'datasets/'")
    df = pd.read_csv(files[0])

    # Features & target
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, le.classes_


# ---------------------------
# Training Functions
# ---------------------------
def train_ann(X_train, X_test, y_train, y_test):
    num_classes = len(np.unique(y_train))

    # One-hot encoding
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train_cat,
              validation_data=(X_test, y_test_cat),
              epochs=100,
              batch_size=10,
              callbacks=[early_stop],
              verbose=0)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    return {
        'Model': 'ANN',
        'Accuracy': accuracy_score(y_test, y_pred_classes),
        'Precision': precision_score(y_test, y_pred_classes, average='weighted'),
        'Recall': recall_score(y_test, y_pred_classes, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred_classes, average='weighted')
    }


def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Model': 'KNN',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }


def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Model': 'SVM',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }


# ---------------------------
# Visualization
# ---------------------------
def plot_results(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    results_df.set_index("Model", inplace=True)

    results_df[metrics].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # scores are between 0 and 1
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# ---------------------------
# Benchmark Pipeline
# ---------------------------
if __name__ == "__main__":
    # Step 1: Analyze dataset
    df = analyze_dataset()

    # Step 2: Load and preprocess
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()

    # Step 3: Train & evaluate models
    results = [
        train_ann(X_train, X_test, y_train, y_test),
        train_knn(X_train, X_test, y_train, y_test),
        train_svm(X_train, X_test, y_train, y_test)
    ]

    # Step 4: Compare results
    results_df = pd.DataFrame(results)
    print("\n=== MODEL BENCHMARK RESULTS ===")
    print(results_df)

    # Step 5: Plot results
    plot_results(results_df)
