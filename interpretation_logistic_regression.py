import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def train_and_evaluate_logistic_regression(input_file):
    # Load the data from the CSV file
    df = pd.read_csv(input_file)

    # Separate features (X) and target (y)
    X = df.drop(columns=['Priority'])
    y = df['Priority']

    # Ensure that 'Priority' column is binary, encode it if necessary
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create a Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=5000)

    # Set up 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Store metrics for each fold
    fold_metrics = []

    print("LOGISTIC REGRESSION")
    print("Fold-wise metrics:")

    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Store metrics for the fold
        fold_metrics.append((fold, precision, recall, auc))

        # Print metrics for the fold
        print(f"Fold {fold}: Precision={precision:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")

    # Calculate and print overall metrics
    overall_precision = np.mean([metric[1] for metric in fold_metrics])
    overall_recall = np.mean([metric[2] for metric in fold_metrics])
    overall_auc = np.mean([metric[3] for metric in fold_metrics])

    print("\nOverall metrics:")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"AUC: {overall_auc:.4f}")

    # Generate graph: Metrics evolution across folds
    folds = [metric[0] for metric in fold_metrics]
    precisions = [metric[1] for metric in fold_metrics]
    recalls = [metric[2] for metric in fold_metrics]
    aucs = [metric[3] for metric in fold_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(folds, precisions, marker='o', label='Precision')
    plt.plot(folds, recalls, marker='o', label='Recall')
    plt.plot(folds, aucs, marker='o', label='AUC')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.title('Metrics Evolution Across Folds')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("fold_metrics_evolution_logistic_regression.png")
    plt.close()
    print("Saved graph of metrics evolution across folds as 'fold_metrics_evolution.png'.")

    # Analyze the impact of removing each column
    column_metrics = {'Column': [], 'Precision': [], 'Recall': [], 'AUC': []}
    for column in X.columns:
        X_temp = X.drop(columns=[column])
        X_temp_scaled = scaler.fit_transform(X_temp)

        precision_list, recall_list, auc_list = [], [], []
        for train_idx, test_idx in cv.split(X_temp_scaled, y):
            X_train, X_test = X_temp_scaled[train_idx], X_temp_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            precision_list.append(precision_score(y_test, y_pred))
            recall_list.append(recall_score(y_test, y_pred))
            auc_list.append(roc_auc_score(y_test, y_proba))

        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_auc = np.mean(auc_list)

        # Print metrics for the column removal
        print(f"Column removed: {column}")
        print(f"Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, AUC: {mean_auc:.4f}\n")

        # Store metrics
        column_metrics['Column'].append(column)
        column_metrics['Precision'].append(mean_precision)
        column_metrics['Recall'].append(mean_recall)
        column_metrics['AUC'].append(mean_auc)

    # Generate graph: Column removal impact
    x = np.arange(len(column_metrics['Column']))
    bar_width = 0.25

    plt.figure(figsize=(12, 8))
    plt.bar(x, column_metrics['Precision'], bar_width, label='Precision')
    plt.bar(x + bar_width, column_metrics['Recall'], bar_width, label='Recall')
    plt.bar(x + 2 * bar_width, column_metrics['AUC'], bar_width, label='AUC')
    plt.xlabel('Column Removed')
    plt.ylabel('Metric Value')
    plt.title('Impact of Column Removal on Metrics')
    plt.xticks(x + bar_width, column_metrics['Column'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("column_removal_impact_logistic_regression.png")
    plt.close()
    print("Saved graph of column removal impact as 'column_removal_impact.png'.")

    # Calculate feature importance
    model.fit(X_scaled, y)
    feature_importances = np.abs(model.coef_).mean(axis=0)

    # Generate graph: Feature importance
    plt.figure(figsize=(12, 6))
    plt.barh(X.columns, feature_importances, color='teal')
    plt.xlabel('Feature Importance (Absolute Coefficient)')
    plt.title('Feature Importance from Logistic Regression')
    plt.tight_layout()
    plt.savefig("feature_importance_logistic_regression.png")
    plt.close()
    print("Saved feature importance graph as 'feature_importance.png'.")

# Example usage
input_file = 'dataset.csv'  # Replace with the path to your input CSV file
train_and_evaluate_logistic_regression(input_file)
