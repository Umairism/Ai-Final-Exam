# Data Preprocessing - Feature Engineering and Data Preparation

**Task 2: Preprocessing (15 Marks)**

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Missing Value Analysis](#missing-value-analysis)
4. [Feature Normalization](#feature-normalization)
5. [Train-Test Split](#train-test-split)
6. [Data Persistence](#data-persistence)
7. [Implementation Guide](#implementation-guide)

---

## Overview

Data preprocessing is one of the most critical steps in machine learning. The quality of your data and preprocessing directly impacts model performance. This phase involves cleaning data, scaling features appropriately, and splitting the dataset for proper model evaluation.

**"Garbage in, garbage out"** - Poor preprocessing leads to poor models, regardless of algorithm sophistication.

---

## Objectives

The preprocessing phase accomplishes three main tasks:

1. **Check for missing values** and handle them appropriately
2. **Normalize/Standardize features** to ensure equal contribution
3. **Split dataset** into training (70%) and testing (30%) sets

---

## Missing Value Analysis

### Why Check for Missing Values?

Missing data can:
- ❌ Cause algorithms to crash
- ❌ Introduce bias in the model
- ❌ Reduce dataset size
- ❌ Lead to incorrect conclusions

### Detection Method

```python
import pandas as pd
import numpy as np

def check_missing_values(df):
    """Comprehensive missing value analysis"""
    
    # Count missing values per column
    missing_values = df.isnull().sum()
    
    # Calculate percentage of missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    # Create summary DataFrame
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Count': missing_values.values,
        'Missing Percentage': missing_percentage.values
    })
    
    print(missing_df)
    
    total_missing = missing_values.sum()
    if total_missing == 0:
        print("✓ No missing values found!")
    else:
        print(f"⚠ Total missing values: {total_missing}")
    
    return total_missing
```

### Interpretation

**No Missing Values (total_missing = 0):**
- ✓ Dataset is complete
- ✓ No imputation needed
- ✓ Can proceed to normalization

**Missing Values Present:**
Need to choose a strategy:

1. **Remove Rows:**
   - Use when: Few missing values (<5%)
   - Pros: Simple, no assumptions
   - Cons: Lose data

2. **Remove Columns:**
   - Use when: Feature has >50% missing
   - Pros: Removes unreliable feature
   - Cons: Lose potentially useful information

3. **Imputation:**
   - Mean/Median: For numerical features
   - Mode: For categorical features
   - Forward/Backward fill: For time series
   - ML-based: KNN imputation, iterative imputation

---

## Feature Normalization

### Why Normalization is CRITICAL for KNN

Feature normalization is **absolutely essential** for K-Nearest Neighbors algorithm. Here's why:

### The Problem: Scale Sensitivity

**Example Scenario:**

```
Dataset with two features:
- Feature1: Age (ranges 0-100)
- Feature2: Salary (ranges 0-100,000)

Three data points:
- Person A: [25, 50000]
- Person B: [26, 50000]
- Person C: [25, 60000]

Which is closer to Person A: B or C?
```

**Without Normalization:**
```python
Distance(A, B) = √[(25-26)² + (50000-50000)²] = √[1 + 0] = 1.0
Distance(A, C) = √[(25-25)² + (50000-60000)²] = √[0 + 100,000,000] = 10,000

KNN Conclusion: B is closer (distance = 1)
```

**The salary feature completely dominates** the distance calculation!

**With Normalization:**
```python
After standardization (mean=0, std=1):
- Age: scaled appropriately
- Salary: scaled appropriately

Distance(A, B) ≈ small contribution from age difference
Distance(A, C) ≈ similar contribution from salary difference

Both features contribute proportionally!
```

### Mathematical Foundation

**Euclidean Distance Formula:**

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

When features have different scales:
- Large-scale features dominate the sum
- Small-scale features become irrelevant
- Model essentially ignores smaller-scale features

### Standardization (Z-score Normalization)

**Formula:**

$$z = \frac{x - \mu}{\sigma}$$

Where:
- $x$ = original value
- $\mu$ = mean of feature
- $\sigma$ = standard deviation of feature
- $z$ = standardized value

**Properties:**
- Mean ($\mu$) = 0
- Standard deviation ($\sigma$) = 1
- Preserves distribution shape
- Handles outliers better than min-max scaling

### Implementation

```python
from sklearn.preprocessing import StandardScaler

def normalize_features(df):
    """Standardize features using StandardScaler"""
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    print("Original Features Statistics:")
    print(X.describe())
    
    # Initialize and apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame for convenience
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("\nScaled Features Statistics (Mean ≈ 0, Std ≈ 1):")
    print(X_scaled_df.describe())
    
    return X_scaled, y, scaler
```

### Why StandardScaler?

**StandardScaler vs Other Methods:**

| Method | Formula | When to Use | Pros | Cons |
|--------|---------|-------------|------|------|
| **StandardScaler** | $(x - \mu) / \sigma$ | Most cases, especially with outliers | Preserves outliers, works with negative values | Assumes Gaussian distribution |
| **MinMaxScaler** | $(x - min) / (max - min)$ | When you need specific range [0,1] | Bounded output | Sensitive to outliers |
| **RobustScaler** | $(x - median) / IQR$ | Data with many outliers | Very robust to outliers | May not preserve distribution shape |

**For KNN:** StandardScaler is preferred because:
1. Handles outliers reasonably well
2. Doesn't compress data into fixed range
3. Works well with Euclidean distance
4. Standard practice in industry

### Visualization of Normalization

```python
import matplotlib.pyplot as plt

def visualize_normalization(X_original, X_scaled):
    """Compare before and after normalization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before normalization
    axes[0].boxplot([X_original[col] for col in X_original.columns], 
                    labels=X_original.columns)
    axes[0].set_title('Before Normalization', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # After normalization
    axes[1].boxplot([X_scaled[col] for col in X_scaled.columns], 
                    labels=X_scaled.columns)
    axes[1].set_title('After Standardization', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Standardized Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=300)
    plt.show()
```

**What to Look For:**
- Before: Different scales, different ranges
- After: Similar scales, centered around 0, comparable spreads

### Naive Bayes vs KNN: Scaling Sensitivity

| Aspect | Naive Bayes | KNN |
|--------|-------------|-----|
| **Sensitivity to Scale** | Low | **VERY HIGH** |
| **Why?** | Uses probability distributions | Uses distance calculations |
| **Requires Scaling?** | No (but doesn't hurt) | **YES (MANDATORY)** |
| **Impact if Not Scaled** | Minimal | **Catastrophic performance** |

---

## Train-Test Split

### Why Split the Data?

**The Golden Rule of Machine Learning:**
> "Never test on the data you trained on"

**Reasons:**

1. **Evaluate Generalization:**
   - Training set: Model learns patterns
   - Testing set: Model proves it can generalize
   - Without split: Can't assess real-world performance

2. **Detect Overfitting:**
   - High training accuracy, low test accuracy = overfitting
   - Model memorized training data but can't generalize

3. **Honest Performance Estimate:**
   - Testing set simulates unseen, real-world data
   - Provides unbiased performance metric

### The 70-30 Split

**Training Set: 70%**
- Used to fit the model
- Model learns patterns, relationships, decision boundaries
- Larger = more learning opportunity

**Testing Set: 30%**
- Used to evaluate model
- Never seen during training
- Smaller but large enough for reliable estimates

**Alternative Splits:**
- 80-20: More common in larger datasets
- 60-40: When data is scarce
- Cross-validation: More robust but computationally expensive

### Stratified Split

**What is Stratification?**

Ensures that class distribution is preserved in both training and testing sets.

**Without Stratification:**
```
Original: 50% Class 0, 50% Class 1
Training: 60% Class 0, 40% Class 1  ❌ Imbalanced!
Testing:  30% Class 0, 70% Class 1  ❌ Imbalanced!
```

**With Stratification:**
```
Original: 50% Class 0, 50% Class 1
Training: 50% Class 0, 50% Class 1  ✓ Balanced!
Testing:  50% Class 0, 50% Class 1  ✓ Balanced!
```

### Implementation

```python
from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.3, random_state=42):
    """Split dataset with stratification"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,           # 30% for testing
        random_state=random_state,      # Reproducibility
        stratify=y                      # Preserve class distribution
    )
    
    print(f"Training set: {X_train.shape[0]} samples (70%)")
    print(f"Testing set: {X_test.shape[0]} samples (30%)")
    print(f"Number of features: {X_train.shape[1]}")
    
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts())
    print("\nClass distribution in testing set:")
    print(pd.Series(y_test).value_counts())
    
    return X_train, X_test, y_train, y_test
```

### Random State Parameter

**Why use `random_state=42`?**

1. **Reproducibility:**
   - Same split every time you run the code
   - Allows others to reproduce your results
   - Essential for debugging

2. **Fair Comparison:**
   - Different models tested on same data
   - Results are directly comparable

3. **Scientific Rigor:**
   - Results can be verified
   - Research can be replicated

**Note:** The value 42 is arbitrary (popularized by "Hitchhiker's Guide to the Galaxy"). Any integer works.

---

## Data Persistence

### Why Save Preprocessed Data?

1. **Efficiency:**
   - Don't repeat preprocessing for each model
   - Save computation time

2. **Consistency:**
   - All models use exact same preprocessed data
   - Fair comparison

3. **Reproducibility:**
   - Can restart from preprocessed state
   - Version control of preprocessing steps

### Implementation

```python
import numpy as np
import pickle

def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler):
    """Save all preprocessed components"""
    
    # Save arrays
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save scaler for future use (e.g., new data prediction)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✓ Preprocessed data saved!")
```

### What Gets Saved?

1. **X_train.npy:** Scaled training features
2. **X_test.npy:** Scaled testing features
3. **y_train.npy:** Training labels
4. **y_test.npy:** Testing labels
5. **scaler.pkl:** Fitted StandardScaler object

**Why save the scaler?**
- To transform new, unseen data the same way
- Must use same mean and std from training
- Critical for production deployment

---

## Implementation Guide

### Complete Code

```python
"""
Task 2: Preprocessing (15 Marks)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    # Load dataset
    df = pd.read_csv('complex_binary_dataset.csv')
    
    # Check for missing values
    check_missing_values(df)
    
    # Normalize features
    X_scaled, y, scaler = normalize_features(df)
    
    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Save preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test, scaler)
    
    print("\nPreprocessing complete!")
    print("Next: Train machine learning models")

if __name__ == "__main__":
    main()
```

### Execution Flow

```
1. Load Data
   ↓
2. Check Missing Values
   ↓
3. Separate Features & Target
   ↓
4. Apply StandardScaler
   ↓
5. Split into Train/Test
   ↓
6. Save All Components
   ↓
7. Ready for Model Training!
```

---

## Key Takeaways

### Critical Points

✓ **Missing values must be handled** before modeling  
✓ **Standardization is MANDATORY for KNN** due to distance calculations  
✓ **Train-test split prevents overfitting** and enables honest evaluation  
✓ **Stratification preserves class balance** in both sets  
✓ **Save preprocessed data** for efficiency and consistency  

### Common Mistakes to Avoid

❌ **Scaling before splitting:** Can cause data leakage  
❌ **Fitting scaler on test data:** Test data must be unseen  
❌ **Ignoring class imbalance:** Can bias model  
❌ **Using different scalers:** Breaks consistency  
❌ **Not saving the scaler:** Can't transform new data correctly  

### Correct Order

```
1. Load data
2. Check missing values
3. Split into train/test
4. Fit scaler on training data ONLY
5. Transform training data
6. Transform testing data (using training scaler)
7. Save everything
```

---

## Next Steps

After preprocessing, proceed to:

1. **Naive Bayes Classification:** Train probabilistic classifier
2. **KNN Classification:** Train distance-based classifier
3. **Model Comparison:** Evaluate and compare performance

---

**File Reference:** `2_preprocessing.py`  
**Previous:** [Data_Understanding.md](Data_Understanding.md)  
**Next:** [Naive_Bayes.md](Naive_Bayes.md)
