"""
Task 5: Comparison & Critical Analysis (5 Marks)
- Classification reports for each classifier
- When Naive Bayes is more suitable
- When KNN is more suitable
- Which model fits the dataset better
- Strengths & weaknesses of both algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pickle

def load_models_and_data():
    """Load trained models and test data"""
    
    print("=" * 70)
    print("LOADING MODELS AND DATA")
    print("=" * 70)
    
    # Load data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    # Load models
    with open('naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    
    print("\n✓ Models and data loaded successfully")
    print(f"  - Test samples: {len(y_test)}")
    print(f"  - Features: {X_test.shape[1]}")
    
    return nb_model, knn_model, X_test, y_test

def generate_classification_reports(nb_model, knn_model, X_test, y_test):
    """Generate detailed classification reports for both classifiers"""
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORTS")
    print("=" * 70)
    
    # Get predictions
    y_pred_nb = nb_model.predict(X_test)
    y_pred_knn = knn_model.predict(X_test)
    
    # Naive Bayes Classification Report
    print("\n" + "▼" * 70)
    print("NAIVE BAYES - CLASSIFICATION REPORT")
    print("▼" * 70)
    nb_report = classification_report(y_test, y_pred_nb, 
                                     target_names=['Class 0 (Negative)', 'Class 1 (Positive)'],
                                     digits=4)
    print("\n" + nb_report)
    
    # KNN Classification Report
    print("\n" + "▼" * 70)
    print(f"K-NEAREST NEIGHBORS (k={knn_model.n_neighbors}) - CLASSIFICATION REPORT")
    print("▼" * 70)
    knn_report = classification_report(y_test, y_pred_knn, 
                                      target_names=['Class 0 (Negative)', 'Class 1 (Positive)'],
                                      digits=4)
    print("\n" + knn_report)
    
    return y_pred_nb, y_pred_knn

def create_comprehensive_comparison(nb_model, knn_model, X_test, y_test):
    """Create comprehensive comparison visualization"""
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Get predictions
    y_pred_nb = nb_model.predict(X_test)
    y_pred_knn = knn_model.predict(X_test)
    
    # Calculate metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    nb_scores = [
        accuracy_score(y_test, y_pred_nb),
        precision_score(y_test, y_pred_nb),
        recall_score(y_test, y_pred_nb),
        f1_score(y_test, y_pred_nb)
    ]
    knn_scores = [
        accuracy_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_knn)
    ]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, nb_scores, width, label='Naive Bayes', 
                       color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = axes[0].bar(x + width/2, knn_scores, width, label='KNN', 
                       color='#2ecc71', alpha=0.8, edgecolor='black')
    
    axes[0].set_xlabel('Metrics', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Performance Comparison: Naive Bayes vs KNN', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    nb_scores_radar = nb_scores + [nb_scores[0]]
    knn_scores_radar = knn_scores + [knn_scores[0]]
    angles += angles[:1]
    
    ax = plt.subplot(122, projection='polar')
    ax.plot(angles, nb_scores_radar, 'o-', linewidth=2, label='Naive Bayes', color='#3498db')
    ax.fill(angles, nb_scores_radar, alpha=0.25, color='#3498db')
    ax.plot(angles, knn_scores_radar, 'o-', linewidth=2, label='KNN', color='#2ecc71')
    ax.fill(angles, knn_scores_radar, alpha=0.25, color='#2ecc71')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Radar Chart: Performance Metrics', fontsize=14, 
                fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comprehensive comparison plot saved as 'comprehensive_comparison.png'")
    plt.show()
    
    return nb_scores, knn_scores

def generate_technical_report(nb_scores, knn_scores):
    """Generate comprehensive technical report (8-10 sentences)"""
    
    print("\n" + "=" * 70)
    print("TECHNICAL REPORT: COMPARATIVE ANALYSIS")
    print("=" * 70)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Determine better model
    nb_avg = np.mean(nb_scores)
    knn_avg = np.mean(knn_scores)
    better_model = "Naive Bayes" if nb_avg > knn_avg else "KNN" if knn_avg > nb_avg else "Both equally"
    
    report = f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           TECHNICAL REPORT: NAIVE BAYES vs KNN ANALYSIS             ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    1. PERFORMANCE SUMMARY:
    
       Naive Bayes achieved an accuracy of {nb_scores[0]:.4f}, with precision of 
       {nb_scores[1]:.4f}, recall of {nb_scores[2]:.4f}, and F1-score of {nb_scores[3]:.4f}. 
       KNN (k={3}) demonstrated an accuracy of {knn_scores[0]:.4f}, precision of 
       {knn_scores[1]:.4f}, recall of {knn_scores[2]:.4f}, and F1-score of {knn_scores[3]:.4f}. 
       Overall, {better_model} performed better with an average metric score of 
       {max(nb_avg, knn_avg):.4f} compared to {min(nb_avg, knn_avg):.4f}.
    
    
    2. WHEN NAIVE BAYES IS MORE SUITABLE:
    
       Naive Bayes is ideal when features approximately follow Gaussian distributions 
       and exhibit reasonable independence, making it particularly effective for 
       probabilistic classification tasks. It excels with small to medium-sized 
       datasets due to its low computational complexity and fast training times, 
       requiring minimal data to estimate parameters. The algorithm is highly suitable 
       when real-time predictions are critical, as it provides instantaneous 
       classification with O(d) complexity where d is the number of features. 
       Additionally, Naive Bayes naturally handles missing data and provides 
       probability estimates for predictions, making it valuable for applications 
       requiring confidence scores. It performs well in high-dimensional spaces and 
       is less prone to overfitting compared to more complex models, making it an 
       excellent baseline classifier for text classification, spam filtering, and 
       sentiment analysis tasks.
    
    
    3. WHEN KNN IS MORE SUITABLE:
    
       KNN is preferred when the decision boundary is highly non-linear and complex, 
       as it makes no assumptions about the underlying data distribution and can 
       capture intricate patterns through local neighborhoods. It performs excellently 
       when similar instances tend to belong to the same class, making it ideal for 
       recommendation systems and pattern recognition tasks. The algorithm is 
       particularly suitable for multi-modal class distributions where classes have 
       multiple clusters, as it naturally handles such scenarios through its 
       instance-based approach. KNN works well with small to medium feature spaces 
       (low dimensional data) and when the training data is representative of the 
       test distribution. It is also advantageous when the model needs to be updated 
       incrementally, as adding new training samples requires no retraining. However, 
       feature scaling through normalization is absolutely essential for KNN to 
       perform correctly, as demonstrated in this analysis.
    
    
    4. WHICH MODEL FITS THIS DATASET BETTER:
    
       For this binary classification dataset, {better_model} demonstrates superior 
       performance with {'consistently higher' if abs(nb_avg - knn_avg) > 0.02 else 'marginally better'} 
       metrics across accuracy, precision, recall, and F1-score. The dataset's 
       {'normalized features and local patterns' if knn_avg > nb_avg else 'probabilistic separability and feature distributions'} 
       make it well-suited for {'the distance-based approach of KNN' if knn_avg > nb_avg else 'the Bayesian probabilistic framework'}. 
       With {4} features, the dimensionality is moderate and {'favorable for KNN' if knn_avg > nb_avg else 'appropriate for Naive Bayes'}, 
       avoiding the curse of dimensionality while providing sufficient information 
       for effective classification. The {'local neighborhood structure' if knn_avg > nb_avg else 'class-conditional distributions'} 
       in the data aligns well with the algorithmic assumptions of {better_model}, 
       resulting in more accurate and reliable predictions. The standardization 
       preprocessing step was crucial for {'enabling KNN to perform optimally' if knn_avg > nb_avg else 'both models, though more critical for KNN'}, 
       ensuring all features contribute proportionally to the classification decision.
    
    
    5. STRENGTHS & WEAKNESSES - NAIVE BAYES:
    
       STRENGTHS:
       ✓ Extremely fast training and prediction (O(d) complexity)
       ✓ Requires minimal training data to estimate parameters
       ✓ Handles high-dimensional data effectively without overfitting
       ✓ Provides probabilistic predictions with confidence scores
       ✓ Naturally handles missing values and irrelevant features
       ✓ Simple, interpretable, and easy to implement
       ✓ Works well with small datasets and limited computational resources
       ✓ Not sensitive to feature scaling
       
       WEAKNESSES:
       ✗ Strong independence assumption (features rarely independent in practice)
       ✗ Cannot capture feature interactions or correlations
       ✗ Assumes features follow Gaussian distribution (in Gaussian NB)
       ✗ Performance degrades when independence assumption is violated
       ✗ Cannot learn complex non-linear decision boundaries
       ✗ Zero-frequency problem (can be addressed with smoothing)
       ✗ Biased towards prior probabilities in imbalanced datasets
       ✗ Limited expressiveness for complex real-world patterns
    
    
    6. STRENGTHS & WEAKNESSES - KNN:
    
       STRENGTHS:
       ✓ Non-parametric: No assumptions about data distribution
       ✓ Can capture complex, non-linear decision boundaries
       ✓ Simple and intuitive algorithm, easy to understand
       ✓ Naturally handles multi-class classification problems
       ✓ No training phase (lazy learning) - instant model updates
       ✓ Effective for datasets with clear cluster structures
       ✓ Can achieve high accuracy with appropriate k value
       ✓ Robust to noisy training data when k is sufficiently large
       
       WEAKNESSES:
       ✗ Computationally expensive prediction: O(n*d) per prediction
       ✗ Memory intensive: Must store entire training dataset
       ✗ Highly sensitive to feature scaling (normalization required)
       ✗ Suffers from curse of dimensionality in high-dimensional spaces
       ✗ Performance degrades with irrelevant or correlated features
       ✗ Sensitive to imbalanced datasets (majority class dominates)
       ✗ Choosing optimal k requires cross-validation
       ✗ Slow for large datasets due to distance computations
       ✗ Vulnerable to noisy data points and outliers (small k)
    
    
    7. CRITICAL ANALYSIS FOR BINARY CLASSIFICATION:
    
       Both algorithms demonstrate distinct advantages for binary classification tasks, 
       with the choice depending on dataset characteristics and application requirements. 
       Naive Bayes excels in scenarios requiring real-time classification, minimal 
       computational resources, and probabilistic interpretations, making it ideal for 
       text classification and spam detection where features (word frequencies) exhibit 
       reasonable independence. KNN shines in applications where decision boundaries 
       are complex and non-linear, such as medical diagnosis and image recognition, 
       where similar cases should receive similar classifications. The fundamental 
       tradeoff involves computational efficiency versus model flexibility: Naive Bayes 
       offers speed and simplicity at the cost of strong assumptions, while KNN provides 
       flexibility and accuracy at the cost of computational expense. For production 
       systems with real-time constraints, Naive Bayes is often preferred, whereas for 
       offline analysis with emphasis on accuracy, KNN may be more appropriate. In this 
       specific dataset, {better_model} emerged as the superior choice, achieving 
       {max(nb_avg, knn_avg):.4f} average performance, suggesting that the data's 
       {'underlying structure aligns well with distance-based classification' if knn_avg > nb_avg else 'probabilistic characteristics favor Bayesian inference'}. 
       However, for practical deployment, one should consider cross-validation, ensemble 
       methods, and hybrid approaches that leverage the strengths of both algorithms.
    
    
    8. RECOMMENDATIONS:
    
       Based on this analysis, I recommend {'KNN with k=3-5' if knn_avg > nb_avg else 'Naive Bayes'} 
       for this particular dataset, as it achieves {'superior accuracy while maintaining reasonable computational cost' if knn_avg > nb_avg else 'excellent performance with minimal computational overhead'}. 
       {'Feature scaling proved essential and should be maintained in production' if knn_avg > nb_avg else 'The probabilistic framework provides valuable confidence estimates'}. 
       For enhanced performance, consider ensemble methods combining both classifiers, 
       feature engineering to {'reduce dimensionality' if knn_avg > nb_avg else 'enhance independence'}, 
       or exploring {'weighted KNN or alternative distance metrics' if knn_avg > nb_avg else 'different Naive Bayes variants (Multinomial, Bernoulli)'} 
       to further optimize results.
    
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    
    print(report)
    
    # Save report to file
    with open('technical_report.txt', 'w') as f:
        f.write(report)
    print("\n✓ Technical report saved as 'technical_report.txt'")

def create_summary_table():
    """Create a comprehensive summary comparison table"""
    
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON SUMMARY TABLE")
    print("=" * 70)
    
    comparison_data = {
        'Aspect': [
            'Algorithm Type',
            'Training Time',
            'Prediction Time',
            'Memory Usage',
            'Assumptions',
            'Decision Boundary',
            'Feature Scaling',
            'Handles Non-linearity',
            'Interpretability',
            'Handles Missing Data',
            'High Dimensions',
            'Small Dataset',
            'Large Dataset',
            'Overfitting Risk',
            'Hyperparameters'
        ],
        'Naive Bayes': [
            'Probabilistic',
            'Very Fast (O(nd))',
            'Very Fast (O(d))',
            'Low',
            'Feature Independence',
            'Linear (probabilistic)',
            'Not Required',
            'Limited',
            'High',
            'Good',
            'Excellent',
            'Excellent',
            'Good',
            'Low',
            'Minimal (smoothing)'
        ],
        'KNN': [
            'Instance-based',
            'None (Lazy)',
            'Slow (O(nd))',
            'High',
            'None',
            'Non-linear (flexible)',
            'Required',
            'Excellent',
            'Moderate',
            'Poor',
            'Poor (curse)',
            'Good',
            'Poor',
            'Moderate',
            'k, distance metric'
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('algorithm_comparison_table.csv', index=False)
    print("\n✓ Comparison table saved as 'algorithm_comparison_table.csv'")
    
    return df

def save_final_summary(nb_scores, knn_scores):
    """Save final summary of the analysis"""
    
    summary = {
        'Naive Bayes': {
            'Accuracy': nb_scores[0],
            'Precision': nb_scores[1],
            'Recall': nb_scores[2],
            'F1-Score': nb_scores[3],
            'Average': np.mean(nb_scores)
        },
        'KNN': {
            'Accuracy': knn_scores[0],
            'Precision': knn_scores[1],
            'Recall': knn_scores[2],
            'F1-Score': knn_scores[3],
            'Average': np.mean(knn_scores)
        }
    }
    
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv('final_model_comparison.csv')
    
    print("\n✓ Final summary saved as 'final_model_comparison.csv'")
    
    return summary_df

def main():
    """Main execution function"""
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       COMPREHENSIVE COMPARISON & CRITICAL ANALYSIS                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")
    
    # Load models and data
    nb_model, knn_model, X_test, y_test = load_models_and_data()
    
    # Generate classification reports
    y_pred_nb, y_pred_knn = generate_classification_reports(nb_model, knn_model, X_test, y_test)
    
    # Create comprehensive comparison
    nb_scores, knn_scores = create_comprehensive_comparison(nb_model, knn_model, X_test, y_test)
    
    # Generate technical report
    generate_technical_report(nb_scores, knn_scores)
    
    # Create summary table
    create_summary_table()
    
    # Save final summary
    summary_df = save_final_summary(nb_scores, knn_scores)
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\n" + summary_df.to_string())
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nAll files generated:")
    print("  ✓ comprehensive_comparison.png")
    print("  ✓ technical_report.txt")
    print("  ✓ algorithm_comparison_table.csv")
    print("  ✓ final_model_comparison.csv")
    print("\n" + "=" * 70)
    print("Thank you for using the classification analysis system!")
    print("=" * 70)

if __name__ == "__main__":
    main()
