import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Load the data
file_path = "dataset.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop(columns=["Priority"])
y = data["Priority"]

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define the model
model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)

fold_metrics = {'Fold': [], 'Precision': [], 'Recall': [], 'AUC': []}
column_metrics = {'Column': [], 'Precision': [], 'Recall': [], 'AUC': []}

# Metrics storage
precision_scores = []
recall_scores = []
auc_scores = []

# Define a custom metrics function
def custom_metrics(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)
    
    # Ensure that one-hot encoding accounts for all classes
    n_classes = y_proba.shape[1]
    y_test_one_hot = np.zeros((len(y_test), n_classes))
    for i, label in enumerate(y_test):
        y_test_one_hot[i, label] = 1
    
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    try:
        auc = roc_auc_score(y_test_one_hot, y_proba, multi_class="ovr", labels=np.arange(n_classes))
    except ValueError:
        # Handle case where AUC cannot be calculated due to missing classes
        auc = np.nan
    return precision, recall, auc

# Perform 10-fold cross-validation
print("Fold-wise metrics:")
for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y_encoded), 1):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    # Check class distribution in the training set
    class_counts = np.bincount(y_train)
    if np.min(class_counts) > smote.k_neighbors + 1:
        # Apply SMOTE only if all classes have enough samples
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    else:
        # Otherwise, skip SMOTE and use original data
        X_resampled, y_resampled = X_train, y_train
    
    # Train the model
    model.fit(X_resampled, y_resampled)
    
    # Evaluate metrics
    precision, recall, auc = custom_metrics(model, X_test, y_test)
    precision_scores.append(precision)
    recall_scores.append(recall)
    auc_scores.append(auc)

    # Store fold metrics
    fold_metrics['Fold'].append(fold)
    fold_metrics['Precision'].append(precision)
    fold_metrics['Recall'].append(recall)
    fold_metrics['AUC'].append(auc)

    # Print metrics for the fold
    print(f"Fold {fold}: Precision={precision:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")

# Display overall metrics, ignoring NaN values for AUC
print("\nOverall metrics:")
print(f"Precision: {np.nanmean(precision_scores):.4f} ± {np.nanstd(precision_scores):.4f}")
print(f"Recall: {np.nanmean(recall_scores):.4f} ± {np.nanstd(recall_scores):.4f}")
print(f"AUC: {np.nanmean(auc_scores):.4f} ± {np.nanstd(auc_scores):.4f}")

# Plot metrics evolution over folds
plt.figure(figsize=(10, 6))
plt.plot(fold_metrics['Fold'], fold_metrics['Precision'], marker='o', label='Precision')
plt.plot(fold_metrics['Fold'], fold_metrics['Recall'], marker='o', label='Recall')
plt.plot(fold_metrics['Fold'], fold_metrics['AUC'], marker='o', label='AUC')
plt.xlabel('Fold')
plt.ylabel('Metric Value')
plt.title('Metrics Evolution Across Folds')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("fold_metrics_evolution_nominal_regression.png")
plt.close()
print("Saved graph of fold metrics evolution as 'fold_metrics_evolution.png'.")

# Analyze the impact of removing each column
for column in X.columns:
    X_temp = X.drop(columns=[column])
    X_temp_scaled = scaler.fit_transform(X_temp)
    
    precision_list, recall_list, auc_list = [], [], []
    for train_index, test_index in skf.split(X_temp_scaled, y_encoded):
        X_train, X_test = X_temp_scaled[train_index], X_temp_scaled[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Apply SMOTE
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Train the model
        model.fit(X_resampled, y_resampled)
        
        # Evaluate metrics
        precision, recall, auc = custom_metrics(model, X_test, y_test)
        precision_list.append(precision)
        recall_list.append(recall)
        auc_list.append(auc)

    # Calculate mean metrics for the column removal
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_auc = np.mean(auc_list)
    
    # Print metrics for the column removal
    print(f"Column removed: {column}")
    print(f"Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, AUC: {mean_auc:.4f}\n")
    
    # Store mean metrics for the column
    column_metrics['Column'].append(column)
    column_metrics['Precision'].append(np.mean(precision_list))
    column_metrics['Recall'].append(np.mean(recall_list))
    column_metrics['AUC'].append(np.mean(auc_list))

# Plot column removal impact
plt.figure(figsize=(12, 8))
x = np.arange(len(column_metrics['Column']))
bar_width = 0.25

plt.bar(x, column_metrics['Precision'], bar_width, label='Precision')
plt.bar(x + bar_width, column_metrics['Recall'], bar_width, label='Recall')
plt.bar(x + 2 * bar_width, column_metrics['AUC'], bar_width, label='AUC')

plt.xlabel('Column Removed')
plt.ylabel('Metric Value')
plt.title('Impact of Column Removal on Metrics')
plt.xticks(x + bar_width, column_metrics['Column'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig("column_removal_impact_nominal_regression.png")
plt.close()
print("Saved graph of column removal impact as 'column_removal_impact.png'.")

# Calculate feature importance
model.fit(X_scaled, y_encoded)
feature_importances = np.abs(model.coef_).mean(axis=0)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.barh(X.columns, feature_importances, color='teal')
plt.xlabel('Feature Importance (Absolute Coefficient)')
plt.title('Feature Importance from Logistic Regression')
plt.tight_layout()
plt.savefig("feature_importance_nominal_regression.png")
plt.close()
print("Saved feature importance graph as 'feature_importance.png'.")
