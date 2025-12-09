# Data Understanding - Exploratory Data Analysis

**Task 1: Data Understanding (5 Marks)**

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [Dataset Exploration](#dataset-exploration)
5. [Visualizations](#visualizations)
6. [Key Observations](#key-observations)
7. [Implementation Details](#implementation-details)

---

## Overview

Data understanding is the foundational step in any machine learning project. This phase involves loading the dataset, exploring its structure, analyzing statistical properties, and visualizing relationships between features. Understanding the data helps identify patterns, detect anomalies, and make informed decisions about preprocessing and modeling strategies.

---

## Objectives

The primary objectives of this data understanding phase are:

1. **Load the dataset** using Python (Pandas)
2. **Display first 10 rows** to understand data structure
3. **Examine dataset shape** (rows and columns)
4. **Calculate summary statistics** for all features
5. **Create visualizations** to understand feature relationships
6. **Analyze class distribution** to check for imbalance

---

## Methodology

### Tools and Libraries

```python
import pandas as pd           # Data manipulation and analysis
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns         # Advanced statistical visualizations
```

### Data Loading Process

The dataset is loaded from a CSV file using Pandas:

```python
df = pd.read_csv('complex_binary_dataset.csv')
```

This creates a DataFrame object that allows efficient data manipulation and analysis.

---

## Dataset Exploration

### 1. First 10 Rows

**Purpose:** Get an initial glimpse of the data structure, feature names, and data types.

**What to Look For:**
- Column names and their meanings
- Data types (numerical, categorical)
- Obvious patterns or anomalies
- Missing values (shown as NaN)
- Range of values

**Code Implementation:**
```python
print(df.head(10))
```

**Interpretation:**
The first 10 rows provide a sample of the dataset, showing:
- Feature columns (feature1, feature2, etc.)
- Target column (label)
- Actual data values and their formats

---

### 2. Dataset Shape

**Purpose:** Understand the size and dimensionality of the dataset.

**Metrics Displayed:**
- **Number of rows:** Total samples/instances in the dataset
- **Number of columns:** Total features + target variable
- **Shape tuple:** (rows, columns)

**Code Implementation:**
```python
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Shape: {df.shape}")
```

**Why It Matters:**
- **Large datasets:** May require sampling or distributed computing
- **Small datasets:** Risk of overfitting, need cross-validation
- **Many features:** Potential for dimensionality reduction
- **Few features:** May need feature engineering

---

### 3. Summary Statistics

**Purpose:** Understand the central tendency, spread, and distribution of numerical features.

**Statistics Calculated:**
- **Count:** Number of non-null values
- **Mean:** Average value
- **Std:** Standard deviation (spread)
- **Min:** Minimum value
- **25%:** First quartile
- **50%:** Median (second quartile)
- **75%:** Third quartile
- **Max:** Maximum value

**Code Implementation:**
```python
print(df.describe())
```

**What to Analyze:**
1. **Range:** Difference between min and max
   - Large ranges suggest need for normalization
   
2. **Standard Deviation:** Measure of variability
   - High std: Data is spread out
   - Low std: Data is clustered around mean
   
3. **Quartiles:** Distribution shape
   - Compare median to mean for skewness
   - Check for outliers beyond 1.5×IQR

4. **Feature Scales:** Different units/magnitudes
   - Critical for distance-based algorithms like KNN

---

### 4. Dataset Information

**Purpose:** Get comprehensive overview of data types and memory usage.

**Information Provided:**
- Column names
- Non-null counts (missing value detection)
- Data types (int64, float64, object)
- Memory usage

**Code Implementation:**
```python
df.info()
```

**Key Checks:**
- ✓ All features are numerical (required for classification)
- ✓ No missing values (or plan to handle them)
- ✓ Target variable has correct type
- ✓ Reasonable memory footprint

---

### 5. Class Distribution

**Purpose:** Analyze the distribution of target classes to detect imbalance.

**Metrics:**
- **Value counts:** Number of samples per class
- **Percentage distribution:** Proportion of each class

**Code Implementation:**
```python
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True) * 100)
```

**Interpretation:**

**Balanced Dataset (50/50):**
- Equal or near-equal class distribution
- No special handling needed
- Standard metrics work well

**Imbalanced Dataset (e.g., 90/10):**
- One class dominates
- May need:
  - Resampling (over/under-sampling)
  - Class weights
  - Different metrics (F1-score, ROC-AUC)
  - Ensemble methods

**Why It Matters:**
- Imbalanced classes can bias models toward majority class
- Accuracy alone is misleading with imbalance
- KNN particularly sensitive to class imbalance

---

## Visualizations

### 1. Scatter Plot: Feature1 vs Feature2

**Purpose:** Visualize the relationship between two features and how they separate classes.

**Code Implementation:**
```python
plt.figure(figsize=(10, 8))

for label in df['label'].unique():
    subset = df[df['label'] == label]
    plt.scatter(subset['feature1'], subset['feature2'], 
               label=f'Class {label}', 
               alpha=0.6,  # Transparency
               s=50,       # Point size
               edgecolors='black',
               linewidths=0.5)

plt.xlabel('Feature 1', fontsize=12, fontweight='bold')
plt.ylabel('Feature 2', fontsize=12, fontweight='bold')
plt.title('Scatter Plot: Feature1 vs Feature2 (Colored by Class Label)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('scatter_plot_feature1_vs_feature2.png', dpi=300)
plt.show()
```

**What to Look For:**

1. **Separability:**
   - Clear clusters → Easy classification
   - Overlapping points → Harder classification
   - Linear separation → Simple models work
   - Non-linear patterns → Need complex models

2. **Patterns:**
   - Linear relationships
   - Circular/elliptical clusters
   - Multiple sub-clusters
   - Outliers far from main groups

3. **Class Distribution:**
   - Uniform spread vs concentrated regions
   - Overlap between classes
   - Class boundaries

**Interpretation for Algorithm Selection:**
- **Linear separation:** Naive Bayes may work well
- **Clustered patterns:** KNN should perform well
- **Complex boundaries:** May need advanced algorithms

---

### 2. Pairplot: All Features

**Purpose:** Comprehensive visualization showing relationships between all feature pairs.

**Code Implementation:**
```python
pairplot = sns.pairplot(df, hue='label', diag_kind='kde', 
                       palette='Set1', plot_kws={'alpha': 0.6, 's': 40})
pairplot.fig.suptitle('Pairplot of All Features by Class Label', 
                      y=1.02, fontsize=16, fontweight='bold')
plt.savefig('pairplot_all_features.png', dpi=300)
plt.show()
```

**Components:**

1. **Diagonal:** Distribution of each feature (KDE plots)
   - Shows if features follow Gaussian distribution
   - Important for Gaussian Naive Bayes assumption

2. **Off-diagonal:** Scatter plots of feature pairs
   - Reveals correlations between features
   - Shows separability from different angles

**Analysis:**

**Feature Correlations:**
- High correlation → Features provide redundant information
- Low correlation → Each feature adds unique value
- Naive Bayes assumes independence (violated by correlation)

**Distribution Shapes:**
- Gaussian (bell-shaped) → Good for Gaussian Naive Bayes
- Skewed → May need transformation
- Bimodal → Multiple clusters within a class

---

## Key Observations

### Statistical Insights

1. **Feature Scales:**
   - Features may have different ranges
   - Normalization required for KNN
   - Less critical for Naive Bayes

2. **Feature Distributions:**
   - Check if approximately Gaussian
   - Affects Naive Bayes performance
   - KNN less sensitive to distribution shape

3. **Class Balance:**
   - Balanced classes simplify analysis
   - Imbalanced classes require special handling
   - Affects choice of evaluation metrics

### Visual Insights

1. **Separability:**
   - How well classes are separated visually
   - Indicates classification difficulty
   - Helps set performance expectations

2. **Cluster Structure:**
   - Presence of clear clusters
   - Multiple modes per class
   - Affects algorithm choice

3. **Outliers:**
   - Extreme values far from main distribution
   - Can affect KNN (pulls neighborhoods)
   - Naive Bayes more robust to outliers

---

## Implementation Details

### Complete Code Structure

```python
"""
Task 1: Data Understanding (5 Marks)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data(filepath):
    """Load and perform initial data exploration"""
    df = pd.read_csv(filepath)
    
    # Display explorations
    print("First 10 rows:", df.head(10))
    print("Shape:", df.shape)
    print("Summary statistics:", df.describe())
    print("Info:", df.info())
    print("Class distribution:", df['label'].value_counts())
    
    return df

def plot_scatter(df):
    """Create scatter plot and pairplot"""
    # Main scatter plot
    plt.figure(figsize=(10, 8))
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        plt.scatter(subset['feature1'], subset['feature2'], 
                   label=f'Class {label}', alpha=0.6, s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Feature1 vs Feature2')
    plt.legend()
    plt.savefig('scatter_plot_feature1_vs_feature2.png', dpi=300)
    plt.show()
    
    # Comprehensive pairplot
    pairplot = sns.pairplot(df, hue='label', diag_kind='kde')
    plt.savefig('pairplot_all_features.png', dpi=300)
    plt.show()

def main():
    filepath = 'complex_binary_dataset.csv'
    df = load_and_explore_data(filepath)
    plot_scatter(df)
    print("Data understanding complete!")

if __name__ == "__main__":
    main()
```

---

## Next Steps

After completing data understanding, proceed to:

1. **Preprocessing:** Handle missing values, normalize features, split data
2. **Model Training:** Train Naive Bayes and KNN classifiers
3. **Evaluation:** Assess model performance
4. **Comparison:** Determine which algorithm performs better

---

## Key Takeaways

✓ **Data understanding is crucial** for successful machine learning  
✓ **Visualizations reveal patterns** not apparent in statistics alone  
✓ **Feature scales matter** for distance-based algorithms  
✓ **Class distribution affects** model training and evaluation  
✓ **Statistical properties guide** algorithm selection and preprocessing decisions  

---

**File Reference:** `1_data_understanding.py`  
**Next Document:** [Preprocessing.md](Preprocessing.md)
