import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    """
    Load the credit card fraud detection dataset
    """
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def basic_eda(df):
    """
    Perform basic exploratory data analysis for credit card fraud
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    
    # Check class distribution
    class_distribution = df['Class'].value_counts(normalize=True)
    
    # Create fraud distribution plot with counts
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Class', data=df)
    plt.title('Distribution of Legitimate vs Fraudulent Transactions')
    plt.xlabel('Class (0: Legitimate, 1: Fraud)')
    plt.ylabel('Number of Transactions')
    
    # Add count labels on top of each bar
    for i in ax.containers:
        ax.bar_label(i, fmt='%d', padding=3)
    
    plt.savefig('outputs/class_distribution.png')
    plt.close()
    
    # Create transaction amount boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.title('Transaction Amount Distribution by Class')
    plt.xlabel('Class (0: Legitimate, 1: Fraud)')
    plt.ylabel('Amount ($)')
    plt.savefig('outputs/amount_boxplot.png')
    plt.close()
    
    # Create separate amount distribution plots for legitimate and fraudulent transactions
    plt.figure(figsize=(15, 5))
    
    # Plot for legitimate transactions (Class 0)
    plt.subplot(1, 2, 1)
    legitimate_amounts = df[df['Class'] == 0]['Amount']
    sns.histplot(data=legitimate_amounts, bins=50)
    plt.title('Amount Distribution - Legitimate Transactions')
    plt.xlabel('Amount ($)')
    plt.ylabel('Count')
    
    # Plot for fraudulent transactions (Class 1)
    plt.subplot(1, 2, 2)
    fraud_amounts = df[df['Class'] == 1]['Amount']
    sns.histplot(data=fraud_amounts, bins=50, color='red')
    plt.title('Amount Distribution - Fraudulent Transactions')
    plt.xlabel('Amount ($)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('outputs/amount_distribution_by_class.png')
    plt.close()
    
    # Add summary statistics
    amount_stats = pd.DataFrame({
        'Legitimate': legitimate_amounts.describe(),
        'Fraudulent': fraud_amounts.describe()
    }).round(2)
    
    print("\nTransaction Amount Statistics:")
    print(amount_stats)
    
    return missing_values, class_distribution

def preprocess_data(df):
    """
    Preprocess the credit card fraud data
    """
    # Create copy of dataframe
    df_processed = df.copy()
    
    # Scale Amount feature
    scaler = StandardScaler()
    df_processed['Amount'] = scaler.fit_transform(df_processed['Amount'].values.reshape(-1, 1))
    
    # Time feature engineering
    df_processed['Time_Hour'] = df_processed['Time'] / 3600  # Convert to hours
    df_processed = df_processed.drop('Time', axis=1)  # Drop original Time column
    
    # Save processed data
    df_processed.to_csv('data/cleaned_data.csv', index=False)
    
    return df_processed

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Load data
    print("Loading credit card fraud dataset...")
    df = load_data('creditcard.csv')
    
    # Perform EDA
    print("\nPerforming exploratory data analysis...")
    missing_values, class_distribution = basic_eda(df)
    print("\nMissing Values:\n", missing_values)
    print("\nClass Distribution:\n", class_distribution)
    print("\nClass distribution plots saved to outputs/class_distribution.png")
    print("Amount distribution plots saved to outputs/amount_distribution_by_class.png")
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_processed = preprocess_data(df)
    print("Preprocessing completed. Cleaned data saved to 'data/cleaned_data.csv'") 