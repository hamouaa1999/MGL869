import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the data
file_path = "dataset_p1.csv"  # Replace with your file path
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
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the model
model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42, k_neighbors=1)

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

# Perform 5-fold cross-validation
for train_index, test_index in skf.split(X_scaled, y_encoded):
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

# Convert scores to arrays
precision_scores = np.array(precision_scores)
recall_scores = np.array(recall_scores)
auc_scores = np.array(auc_scores)

# Display results, ignoring NaN values for AUC
print(f"Precision: {np.nanmean(precision_scores):.4f} ± {np.nanstd(precision_scores):.4f}")
print(f"Recall: {np.nanmean(recall_scores):.4f} ± {np.nanstd(recall_scores):.4f}")
print(f"AUC: {np.nanmean(auc_scores):.4f} ± {np.nanstd(auc_scores):.4f}")
