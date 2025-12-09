"""
Task 2: Preprocessing (15 Marks)
- Check for missing values
- Normalize or standardize the features
- Split the dataset into training (70%) and testing (30%)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def check_missing_values(df):
    """Check for missing values in the dataset"""
    
    print("=" * 60)
    print("CHECKING FOR MISSING VALUES")
    print("=" * 60)
    
    # Count missing values
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percentage': missing_percentage.values
    })
    
    print("\nMissing Values Summary:")
    print("-" * 60)
    print(missing_df)
    
    total_missing = missing_values.sum()
    if total_missing == 0:
        print("\n✓ No missing values found in the dataset!")
    else:
        print(f"\n⚠ Total missing values: {total_missing}")
    
    return total_missing

def explain_normalization():
    """Explain why normalization is required for KNN"""
    
    print("\n" + "=" * 60)
    print("WHY NORMALIZATION IS REQUIRED FOR KNN")
    print("=" * 60)
    
    explanation = """
    Normalization/Standardization is CRITICAL for KNN because:
    
    1. DISTANCE-BASED ALGORITHM:
       - KNN uses distance metrics (e.g., Euclidean distance) to find nearest neighbors
       - Features with larger scales dominate the distance calculation
       
    2. SCALE SENSITIVITY:
       - If Feature1 ranges from 0-1 and Feature2 ranges from 0-1000,
         Feature2 will dominate distance calculations
       - This leads to biased predictions ignoring smaller-scale features
       
    3. EQUAL FEATURE IMPORTANCE:
       - Standardization ensures all features contribute equally
       - Transforms features to have mean=0 and std=1
       
    4. EXAMPLE:
       Point A: [1, 100]
       Point B: [2, 100]
       Point C: [1, 200]
       
       Without normalization: Distance(A,B) ≈ 1, Distance(A,C) ≈ 100
       → Feature2 dominates!
       
       With normalization: Both features contribute proportionally
       
    5. NAIVE BAYES vs KNN:
       - Naive Bayes: Works with probability distributions, less sensitive to scale
       - KNN: Directly uses distances, HIGHLY sensitive to scale
       
    CONCLUSION: Standardization is MANDATORY for KNN to perform correctly!
    """
    
    print(explanation)

def normalize_features(df):
    """Normalize/Standardize the features"""
    
    print("\n" + "=" * 60)
    print("FEATURE NORMALIZATION (STANDARDIZATION)")
    print("=" * 60)
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    print("\nOriginal Features Statistics:")
    print("-" * 60)
    print(X.describe())
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the features
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("\n✓ Features standardized using StandardScaler")
    print("\nScaled Features Statistics (Mean ≈ 0, Std ≈ 1):")
    print("-" * 60)
    print(X_scaled_df.describe())
    
    # Visualize before and after normalization
    visualize_normalization(X, X_scaled_df)
    
    return X_scaled, y, scaler

def visualize_normalization(X_original, X_scaled):
    """Visualize the effect of normalization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original data
    axes[0].boxplot([X_original[col] for col in X_original.columns], 
                    labels=X_original.columns)
    axes[0].set_title('Before Normalization', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Scaled data
    axes[1].boxplot([X_scaled[col] for col in X_scaled.columns], 
                    labels=X_scaled.columns)
    axes[1].set_title('After Standardization', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Standardized Value', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Normalization comparison plot saved as 'normalization_comparison.png'")
    plt.show()

def split_dataset(X, y, test_size=0.3, random_state=42):
    """Split dataset into training (70%) and testing (30%) sets"""
    
    print("\n" + "=" * 60)
    print("SPLITTING DATASET")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nDataset split with test_size={test_size} (Training: {1-test_size:.0%}, Testing: {test_size:.0%})")
    print("-" * 60)
    print(f"Training set size: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"Testing set size: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    print(f"Number of features: {X_train.shape[1]}")
    
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts())
    print("\nClass distribution in testing set:")
    print(pd.Series(y_test).value_counts())
    
    print("\n✓ Stratified split ensures balanced class distribution in both sets")
    
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler):
    """Save preprocessed data for use in subsequent tasks"""
    
    print("\n" + "=" * 60)
    print("SAVING PREPROCESSED DATA")
    print("=" * 60)
    
    # Save as numpy arrays
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save scaler
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n✓ Preprocessed data saved:")
    print("  - X_train.npy")
    print("  - X_test.npy")
    print("  - y_train.npy")
    print("  - y_test.npy")
    print("  - scaler.pkl")

def main():
    """Main execution function"""
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('complex_binary_dataset.csv')
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Check for missing values
    check_missing_values(df)
    
    # Explain why normalization is needed
    explain_normalization()
    
    # Normalize features
    X_scaled, y, scaler = normalize_features(df)
    
    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X_scaled, y)
    
    # Save preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test, scaler)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nKey Steps Completed:")
    print("1. ✓ Missing values checked")
    print("2. ✓ Features normalized (standardized)")
    print("3. ✓ Dataset split (70% train, 30% test)")
    print("4. ✓ Data saved for model training")
    print("\nNext Step: Proceed to Naive Bayes classification (3_naive_bayes.py)")

if __name__ == "__main__":
    main()
