import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate_logistic_regression(input_file):
    # Load the data from the CSV file
    df = pd.read_csv(input_file)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['Priority'])
    y = df['Priority']
    
    # Ensure that 'Priority' column is binary, encode it if necessary (assuming 'Priority' column is categorical)
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Create a Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000)

    # Set up 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Perform cross-validation with predictions
    predicted = cross_val_predict(model, X, y, cv=cv)

    # Calculate metrics
    precision = precision_score(y, predicted)
    recall = recall_score(y, predicted)
    auc = roc_auc_score(y, predicted)

    print("LOGISTIC REGRESSION")

    # Print the metrics
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")

# Example usage
input_file = 'dataset.csv'  # Replace with the path to your input CSV file
train_and_evaluate_logistic_regression(input_file)
