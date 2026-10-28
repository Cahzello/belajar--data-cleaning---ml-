import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

X_train = pd.read_csv('./result/training.csv')
X_test = pd.read_csv('./result/testing.csv')
y_train = pd.read_csv('./result/train_y.csv').squeeze()
y_test = pd.read_csv('./result/test_y.csv').squeeze()

if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

bool_cols_train = X_train.select_dtypes(include=['bool']).columns
bool_cols_test = X_test.select_dtypes(include=['bool']).columns
if len(bool_cols_train) > 0:
    X_train[bool_cols_train] = X_train[bool_cols_train].astype(int)
if len(bool_cols_test) > 0:
    X_test[bool_cols_test] = X_test[bool_cols_test].astype(int)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print('=== Data Shape ===')
print('X_train:', X_train.shape, '| y_train:', y_train.shape)
print('X_test:', X_test.shape, '| y_test:', y_test.shape)
print('\n=== Class Distribution ===')
print('y_train:\n', y_train.value_counts(normalize=True))
print('y_test:\n', y_test.value_counts(normalize=True))

clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print('\n=== Confusion Matrix ===')
print(confusion_matrix(y_test, y_pred))

print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred))

print('=== Metrics ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall   : {recall_score(y_test, y_pred):.4f}')
print(f'F1-Score : {f1_score(y_test, y_pred):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}')

z_train = clf.decision_function(X_train)

df_logit = X_train.copy()
df_logit['logit'] = z_train
df_logit['Churn'] = y_train.values

df_logit_sample = df_logit.sample(n=10000, random_state=42)

numeric_for_plot = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age']

for col in numeric_for_plot:
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df_logit_sample,
        x=col,
        y='logit',
        scatter_kws={'alpha': 0.3, 's': 10},
        line_kws={'color': 'red'}
    )
    plt.title(f'Scatter Plot Logit(\u0177) vs {col}')
    plt.xlabel(col)
    plt.ylabel('Logit (Linear Predictor)')
    plt.tight_layout()
    fname = f'./result/logit_vs_{col.replace(" ", "_").lower()}.png'
    plt.savefig(fname, dpi=150)
    plt.close()



