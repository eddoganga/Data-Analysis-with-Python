## Project: Churn prediction for Sprint
 Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that.

So, if you were in charge of predicting customer churn how would you go about using machine learning to make a good guess about which customers might leave? Like, what steps would you take to create a machine learning model that can predict if someone's going to leave or not?


## Define Churn Event:
A churn event typically refers to a specific event or action taken by a customer that signifies their decision to discontinue their relationship with the bank or to cease using one or more of the bank's services. 

## Data Collection:
Gather historical customer data, including demographic information, usage patterns, contract details, customer service interactions, and churn history.
Ensure the data is clean, relevant, and well-structured.

```
import pandas as pd
df = pd.read_csv('data.csv')
```

## Data Preprocessing:
Handle missing data by imputing or removing it as appropriate.
Encode categorical variables using techniques like one-hot encoding or label encoding.
Scale numerical features to ensure they have similar scales 

```
df.info()
#handle missing data
df.dropna(inplace=True)
#encode categorical variables
df = pd.get_dummies(df, columns=['gender'], drop_first=True)
#scaling numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['data_usage', 'voice_usage']] = scaler.fit_transform(df[['data_usage', 'voice_usage']])
```

## Feature Engineering:
Create new features if they can provide valuable insights. For example, you can calculate customer tenure, average usage, or loyalty program participation.
Feature selection techniques such as recursive feature elimination or feature importance analysis can help identify the most relevant features.

```
# Calculate customer tenure in months 
df['tenure'] = df['contract_length']

# Calculate average usage of all services
df['average_usage'] = (df['data_usage'] + df['voice_usage'] + df['roaming_usage']) / 3

# Calculate total usage of all services
df['total_usage'] = df['data_usage'] + df['voice_usage'] + df['roaming_usage']

# Create a binary 'high_usage' feature based on a threshold
threshold = 200  
df['high_usage'] = (df['total_usage'] > threshold).astype(int)

# Convert 'churn' to integer (0 for False, 1 for True) if needed
df['churn'] = df['churn'].astype(int)
```

## Model Selection:
Choose appropriate machine learning algorithms for classification tasks. Common choices include logistic regression, decision trees, random forests, gradient boosting, and neural networks.
Consider using ensemble methods to improve model performance.
```
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['churn']), df['churn'], test_size=0.2, random_state=42)

# Create and train a Decision Tree model
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# Evaluate the model on the test set
tree_accuracy = tree_model.score(X_test, y_test)

```

## Model Training:
Train the selected models on the training data.
Use techniques like cross-validation to tune hyperparameters and prevent overfitting.By performing hyperparameter tuning for the Decision Tree model using cross-validation, you can find the best max_depth value to prevent overfitting and optimize model performance.
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)  # Increase max_iter to a suitable value

# Define hyperparameters to tune
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

# Create a GridSearchCV object with cross-validation
grid_search_logistic = GridSearchCV(logistic_model, param_grid, cv=5, scoring='accuracy')

# Fit the model and perform hyperparameter tuning on scaled data
grid_search_logistic.fit(X_train_scaled, y_train)

# Get the best hyperparameters and the best model
best_params_logistic = grid_search_logistic.best_params_
best_model_logistic = grid_search_logistic.best_estimator_

# Evaluate the best model on the test set
logistic_accuracy = best_model_logistic.score(X_test_scaled, y_test)
print("Logistic Regression Accuracy:", logistic_accuracy)
```

## Model Evaluation:
Evaluate model performance using appropriate metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
Consider the business context and choose the evaluation metric that aligns with Sprint's goals. For churn prediction, often recall (to minimize false negatives) and ROC-AUC (to measure overall model performance) are crucial.
```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Evaluate Logistic Regression model
logistic_y_pred = best_model_logistic.predict(X_test_scaled)

logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
logistic_precision = precision_score(y_test, logistic_y_pred)
logistic_recall = recall_score(y_test, logistic_y_pred)
logistic_f1 = f1_score(y_test, logistic_y_pred)
logistic_roc_auc = roc_auc_score(y_test, best_model_logistic.predict_proba(X_test_scaled)[:, 1])

# Print the metrics for Logistic Regression
print("Logistic Regression Metrics:")
print(f"Accuracy: {logistic_accuracy:.2f}")
print(f"Precision: {logistic_precision:.2f}")
print(f"Recall: {logistic_recall:.2f}")
print(f"F1-Score: {logistic_f1:.2f}")
print(f"ROC-AUC: {logistic_roc_auc:.2f}")

# ROC curve for Logistic Regression
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, best_model_logistic.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr_logistic, tpr_logistic, label="Logistic Regression (AUC = {:.2f})".format(logistic_roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```

## Model Interpretability:
Understand why the model makes certain predictions. This can be important for identifying actionable insights to reduce churn.
Techniques like SHAP values, LIME, or feature importance can help explain model predictions.

## Deployment:
Deploy the trained model into the production environment, making it ready for real-time predictions.
Set up a feedback loop to continuously update the model with new data and retrain it periodically.

## Monitoring:
Continuously monitor the model's performance in production to ensure it remains accurate over time.
Implement alerts for significant deviations in churn predictions.

## Actionable Insights:
Translate model predictions into actionable strategies for retaining customers. For instance, target high-risk customers with personalized offers, discounts, or improved customer service.

## Iterate and Improve:
Regularly review and update the model to adapt to changing customer behavior and business conditions.
Compliance:

Ensure that your model complies with data privacy regulations and ethical considerations when using customer data.
