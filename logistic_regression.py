import pandas as pd
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

# Load cleaned data (features) and aligned targets (y)
X_train = pd.read_csv('./result/training.csv')
X_test = pd.read_csv('./result/testing.csv')
y_train = pd.read_csv('./result/train_y.csv').squeeze()
y_test = pd.read_csv('./result/test_y.csv').squeeze()

# Ensure target is 1D array
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

# Convert boolean columns to int (from get_dummies)
bool_cols_train = X_train.select_dtypes(include=['bool']).columns
bool_cols_test = X_test.select_dtypes(include=['bool']).columns
if len(bool_cols_train) > 0:
    X_train[bool_cols_train] = X_train[bool_cols_train].astype(int)
if len(bool_cols_test) > 0:
    X_test[bool_cols_test] = X_test[bool_cols_test].astype(int)

# Ensure numeric types
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

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluation
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

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': clf.coef_[0],
    'Abs_Coefficient': abs(clf.coef_[0])
}).sort_values(by='Abs_Coefficient', ascending=False)

print('\n=== Feature Importance (Top 15 by |Coefficient|) ===')
print(coef_df.head(15).to_string(index=False))
print(f'\nIntercept: {clf.intercept_[0]:.6f}')

# === Scatter Plot: Logit vs Fitur Numerik ===
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs('./result', exist_ok=True)

    # Hitung nilai logit (linear predictor) menggunakan decision_function
    z_train = clf.decision_function(X_train)

    # Gabungkan data untuk plotting
    df_logit = X_train.copy()
    df_logit['logit'] = z_train
    df_logit['Churn'] = y_train.values

    # Sampling untuk performa visualisasi
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
        print(f'Saved: {fname}')

    print('\n=== Logit vs Fitur scatter plots selesai ===')
except Exception as e:
    print(f'Gagal membuat scatter plot logit: {e}')

