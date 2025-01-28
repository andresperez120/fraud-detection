import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def load_processed_data(file_path):
    """
    Load the processed data
    """
    return pd.read_csv(file_path)

def train_model(X_train, y_train):
    """
    Train the XGBoost model with parameters suitable for imbalanced fraud detection
    """
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),  # Handle class imbalance
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, X_train):
    """
    Evaluate the model and create visualizations
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Credit Card Fraud Detection')
    plt.legend(loc="lower right")
    plt.savefig('outputs/roc_curve.png')
    plt.close()
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features for Fraud Detection')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    # Print classification report
    report = classification_report(y_test, y_pred)
    
    return report

if __name__ == "__main__":
    print("Loading processed credit card fraud data...")
    df = load_processed_data('data/cleaned_data.csv')
    
    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining XGBoost model...")
    model = train_model(X_train, y_train)
    
    print("\nEvaluating model performance...")
    report = evaluate_model(model, X_test, y_test, X_train)
    print("\nClassification Report:\n", report)
    
    print("\nSaving model...")
    with open('outputs/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to 'outputs/xgboost_model.pkl'") 