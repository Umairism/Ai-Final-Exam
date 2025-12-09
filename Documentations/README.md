# Naive Bayes and KNN Classification Project

## 📋 Project Overview

This project implements and compares **Naive Bayes** and **K-Nearest Neighbors (KNN)** classification algorithms on a binary classification dataset. The project is organized into 5 separate tasks, each addressing specific aspects of the machine learning pipeline.

## 🎯 Objectives

1. **Data Understanding** - Load, explore, and visualize the dataset
2. **Preprocessing** - Handle missing values, normalize features, and split data
3. **Naive Bayes Classification** - Train and evaluate Gaussian Naive Bayes
4. **KNN Classification** - Train KNN with different k values and compare
5. **Comparison & Analysis** - Generate comprehensive technical report

## 📁 Project Structure

```
Ai Final Exam/
├── complex_binary_dataset.csv          # Input dataset
├── 1_data_understanding.py             # Task 1: Data exploration
├── 2_preprocessing.py                  # Task 2: Data preprocessing
├── 3_naive_bayes.py                    # Task 3: Naive Bayes classifier
├── 4_knn_classification.py             # Task 4: KNN classifier
├── 5_comparison_analysis.py            # Task 5: Comparative analysis
├── run_all_tasks.py                    # Master script to run all tasks
└── README.md                           # This file
```

## 🚀 Getting Started

### Prerequisites

Install required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Run All Tasks at Once (Recommended)

```bash
python run_all_tasks.py
```

This will execute all 5 tasks sequentially and generate all outputs.

#### Option 2: Run Tasks Individually

```bash
# Task 1: Data Understanding
python 1_data_understanding.py

# Task 2: Preprocessing
python 2_preprocessing.py

# Task 3: Naive Bayes
python 3_naive_bayes.py

# Task 4: KNN Classification
python 4_knn_classification.py

# Task 5: Comparison & Analysis
python 5_comparison_analysis.py
```

⚠️ **Important**: Tasks must be run in order as each depends on outputs from previous tasks.

## 📊 Task Breakdown

### Task 1: Data Understanding (5 Marks)
- ✅ Load dataset using Pandas
- ✅ Display first 10 rows, shape, and summary statistics
- ✅ Plot scatter plot of feature1 vs feature2 colored by class label
- ✅ Generate comprehensive pairplot for all features

**Outputs:**
- `scatter_plot_feature1_vs_feature2.png`
- `pairplot_all_features.png`

### Task 2: Preprocessing (15 Marks)
- ✅ Check for missing values
- ✅ Normalize/standardize features using StandardScaler
- ✅ Explain why normalization is required for KNN
- ✅ Split dataset: 70% training, 30% testing (stratified)

**Outputs:**
- `normalization_comparison.png`
- `X_train.npy`, `X_test.npy`
- `y_train.npy`, `y_test.npy`
- `scaler.pkl`

### Task 3: Naive Bayes Classification (10 Marks)
- ✅ Train Gaussian Naive Bayes classifier
- ✅ Predict class labels for test set
- ✅ Evaluate: Accuracy, Precision, Recall, F1-score
- ✅ Generate confusion matrix
- ✅ Interpret results with detailed analysis

**Outputs:**
- `confusion_matrix_naive_bayes.png`
- `naive_bayes_model.pkl`
- `naive_bayes_results.npy`

### Task 4: KNN Classification (10 Marks)
- ✅ Train KNN with k=3 using Euclidean distance
- ✅ Evaluate with same metrics as Naive Bayes
- ✅ Compare performance for k=3, 5, 7
- ✅ Discuss impact of increasing k
- ✅ Compare KNN vs Naive Bayes

**Outputs:**
- `confusion_matrix_knn_k3.png`
- `knn_k_comparison.png`
- `knn_model.pkl`
- `knn_k_comparison_results.csv`

### Task 5: Comparison & Critical Analysis (5 Marks)
- ✅ Classification reports for both classifiers
- ✅ When Naive Bayes is more suitable
- ✅ When KNN is more suitable
- ✅ Which model fits the dataset better
- ✅ Strengths & weaknesses analysis (8-10 sentences)
- ✅ Comprehensive technical report

**Outputs:**
- `comprehensive_comparison.png`
- `technical_report.txt`
- `algorithm_comparison_table.csv`
- `final_model_comparison.csv`

## 📈 Generated Outputs

### Visualizations (PNG files)
1. `scatter_plot_feature1_vs_feature2.png` - Feature space visualization
2. `pairplot_all_features.png` - Comprehensive feature relationships
3. `normalization_comparison.png` - Before/after standardization
4. `confusion_matrix_naive_bayes.png` - NB confusion matrix
5. `confusion_matrix_knn_k3.png` - KNN confusion matrix
6. `knn_k_comparison.png` - Performance across different k values
7. `comprehensive_comparison.png` - Final model comparison

### Models (PKL files)
- `naive_bayes_model.pkl` - Trained Naive Bayes model
- `knn_model.pkl` - Best KNN model
- `scaler.pkl` - Feature scaler for preprocessing

### Data Files (NPY files)
- `X_train.npy`, `X_test.npy` - Feature matrices
- `y_train.npy`, `y_test.npy` - Label vectors
- `naive_bayes_results.npy` - NB performance metrics

### Reports (TXT/CSV files)
- `technical_report.txt` - Comprehensive technical analysis
- `algorithm_comparison_table.csv` - Algorithm comparison
- `final_model_comparison.csv` - Final metrics comparison
- `knn_k_comparison_results.csv` - K-value comparison data

## 🔍 Key Findings

### Why Normalization is Critical for KNN
- **Distance-based algorithm**: KNN uses Euclidean distance to find neighbors
- **Scale sensitivity**: Features with larger scales dominate distance calculations
- **Equal contribution**: Standardization ensures all features contribute equally
- **Example**: Without normalization, a feature ranging 0-1000 will dominate one ranging 0-1

### Model Comparison
The analysis compares both algorithms across multiple dimensions:
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Computational Complexity**: Training and prediction time
- **Assumptions**: Feature independence vs. no assumptions
- **Suitability**: When to use each algorithm

## 💡 Algorithm Insights

### Naive Bayes
**Best For:**
- Small to medium datasets
- Real-time predictions
- High-dimensional data
- Probabilistic interpretations
- Text classification

**Limitations:**
- Assumes feature independence
- Limited to linear boundaries
- Requires Gaussian distribution (Gaussian NB)

### K-Nearest Neighbors
**Best For:**
- Non-linear decision boundaries
- Multi-modal distributions
- Local pattern recognition
- No distribution assumptions

**Limitations:**
- Computationally expensive predictions
- Memory intensive
- Sensitive to feature scaling
- Curse of dimensionality

## 📚 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🛠️ Technical Details

### Feature Scaling
- **Method**: StandardScaler (z-score normalization)
- **Formula**: z = (x - μ) / σ
- **Result**: Mean = 0, Standard Deviation = 1

### Train-Test Split
- **Split Ratio**: 70% train, 30% test
- **Method**: Stratified split (preserves class distribution)
- **Random State**: 42 (for reproducibility)

### Evaluation Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

## 🎓 Learning Outcomes

After completing this project, you will understand:
1. ✅ Complete machine learning pipeline from data loading to model evaluation
2. ✅ Importance of data preprocessing and normalization
3. ✅ Differences between probabilistic and instance-based learning
4. ✅ How to evaluate and compare classification models
5. ✅ When to choose Naive Bayes vs KNN for different problems

## 📝 Notes

- All scripts include detailed comments and explanations
- Visualizations are saved automatically with high DPI (300)
- Models can be reloaded using pickle for future predictions
- Results are reproducible (fixed random seeds)

## 🤝 Contributing

This is an academic project. Feel free to:
- Experiment with different k values for KNN
- Try other Naive Bayes variants (Multinomial, Bernoulli)
- Add more evaluation metrics (ROC-AUC, MCC)
- Implement cross-validation
- Test ensemble methods

## 📧 Contact

For questions or issues, please refer to the inline documentation in each script file.

---

**Created as part of AI Final Exam Project**

*Binary Classification using Naive Bayes and K-Nearest Neighbors*
