import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv('./data/ecommerce_customer_data_large.csv')

df = df.drop_duplicates()
df = df.drop(columns=["Age"])
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"])
df['Returns'] = df['Returns'].fillna(0)
df["Total Purchase Amount"] = df["Product Price"] * df["Quantity"]

df = df.drop(columns=["Customer ID", "Customer Name", "Purchase Date"])
X = df.drop(columns=["Churn"])
y = df["Churn"]

train_df = X.sample(frac=0.7, random_state=42).copy()
test_df = X.drop(train_df.index).copy()

train_y = y.loc[train_df.index]
test_y = y.loc[test_df.index]

numeric_features = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age']

for col in numeric_features:
    q1 = train_df[col].quantile(0.25)
    q3 = train_df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    train_df = train_df[(train_df[col] >= lower_bound) & (train_df[col] <= upper_bound)]
    test_df = test_df[(test_df[col] >= lower_bound) & (test_df[col] <= upper_bound)]

train_y = train_y.loc[train_df.index]
test_y = test_y.loc[test_df.index]

for col in numeric_features:
    mean_val = train_df[col].mean()
    std_val = train_df[col].std()
    train_df[col] = (train_df[col] - mean_val) / std_val
    test_df[col] = (test_df[col] - mean_val) / std_val
    
categorical_features = ['Product Category', 'Payment Method', 'Gender']

train_encoded = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)

train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

print("Jumlah baris sebelum hapus outlier:", len(df))
print("Jumlah baris setelah hapus outlier:", len(train_encoded) + len(test_encoded))
print("Ukuran data train:", train_encoded.shape)
print("Ukuran data test:", test_encoded.shape)
print("\n5 data setelah diproses:")
print(train_encoded.head())
