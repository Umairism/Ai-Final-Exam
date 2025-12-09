"""
Task 1: Data Understanding (5 Marks)
- Load the dataset using Python (Pandas)
- Display first 10 rows, dataset shape, summary statistics
- Plot scatter plot of feature1 vs feature2 colored by class label
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_explore_data(filepath):
    """Load and perform initial data exploration"""
    
    # Load the dataset
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    df = pd.read_csv(filepath)
    
    # Display first 10 rows
    print("\n1. FIRST 10 ROWS OF THE DATASET:")
    print("-" * 60)
    print(df.head(10))
    
    # Display dataset shape
    print("\n2. DATASET SHAPE:")
    print("-" * 60)
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Shape: {df.shape}")
    
    # Display summary statistics
    print("\n3. SUMMARY STATISTICS:")
    print("-" * 60)
    print(df.describe())
    
    # Display data types and info
    print("\n4. DATASET INFO:")
    print("-" * 60)
    print(df.info())
    
    # Display class distribution
    print("\n5. CLASS LABEL DISTRIBUTION:")
    print("-" * 60)
    print(df['label'].value_counts())
    print(f"\nClass Balance:")
    print(df['label'].value_counts(normalize=True) * 100)
    
    return df

def plot_scatter(df):
    """Plot scatter plot of feature1 vs feature2 colored by class label"""
    
    print("\n" + "=" * 60)
    print("CREATING SCATTER PLOT")
    print("=" * 60)
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with different colors for each class
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        plt.scatter(subset['feature1'], subset['feature2'], 
                   label=f'Class {label}', 
                   alpha=0.6, 
                   s=50,
                   edgecolors='black',
                   linewidths=0.5)
    
    plt.xlabel('Feature 1', fontsize=12, fontweight='bold')
    plt.ylabel('Feature 2', fontsize=12, fontweight='bold')
    plt.title('Scatter Plot: Feature1 vs Feature2 (Colored by Class Label)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('scatter_plot_feature1_vs_feature2.png', dpi=300, bbox_inches='tight')
    print("\n✓ Scatter plot saved as 'scatter_plot_feature1_vs_feature2.png'")
    
    plt.show()
    
    # Create additional visualization - pairplot for all features
    print("\nCreating comprehensive pairplot for all features...")
    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(df, hue='label', diag_kind='kde', 
                           palette='Set1', plot_kws={'alpha': 0.6, 's': 40})
    pairplot.fig.suptitle('Pairplot of All Features by Class Label', 
                          y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pairplot_all_features.png', dpi=300, bbox_inches='tight')
    print("✓ Pairplot saved as 'pairplot_all_features.png'")
    plt.show()

def main():
    """Main execution function"""
    # File path
    filepath = 'complex_binary_dataset.csv'
    
    # Load and explore data
    df = load_and_explore_data(filepath)
    
    # Create scatter plot
    plot_scatter(df)
    
    print("\n" + "=" * 60)
    print("DATA UNDERSTANDING COMPLETE!")
    print("=" * 60)
    print("\nKey Observations:")
    print("1. Dataset loaded successfully")
    print("2. Exploratory analysis completed")
    print("3. Visualizations created and saved")
    print("\nNext Step: Proceed to preprocessing (2_preprocessing.py)")

if __name__ == "__main__":
    main()
