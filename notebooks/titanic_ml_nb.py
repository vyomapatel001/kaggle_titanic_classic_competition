# titanic_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load training data
train_df = pd.read_csv('../data/train.csv')

# Preprocess training data
def preprocess(df):
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True, errors='ignore')
    return df

train_processed = preprocess(train_df)

# Prepare features and target
X = train_processed.drop('Survived', axis=1)
y = train_processed['Survived']

# Split for evaluation (optional)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_valid)
print('Validation Accuracy:', accuracy_score(y_valid, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_valid, y_pred))

# Load test data
test_df = pd.read_csv('../data/test.csv')
passenger_ids = test_df['PassengerId']

# Preprocess test data
test_processed = preprocess(test_df)

# Predict on test data
test_predictions = model.predict(test_processed)

# Prepare submission
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': test_predictions
})

# Save to CSV
submission.to_csv('../data/submission.csv', index=False)

print('submission.csv file created successfully.')
