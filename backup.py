import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv('./data/ecommerce_customer_data_large.csv')

# 1. Menghapus data yang duplikat
df = df.drop_duplicates()

# 2. Menghapus kolom umur karena ada 2 kolom yang merepresentasikan data sama
df = df.drop(columns=["Age"])

# 3. Mengganti tipe data purchase date ke datetime dari string
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"])

# 4. Mengisi data yang kosong pada kolom Returns
df['Returns'] = df['Returns'].fillna(0)

# 5. Menghitung kembali total purchase karena ada kekeliruan pada data
df["Total Purchase Amount"] = df["Product Price"] * df["Quantity"]

# 6. Menghapus outlier pada beberapa kolom numerik menggunakan metode IQR
numeric_cols_to_check = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age', 'Returns']

for col in numeric_cols_to_check:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


# ==========================================
# TAHAP PRA-PEMROSESAN MANUAL (TANPA SKLEARN)
# ==========================================

# 7. Memisahkan Fitur (X) dan Target (y)
X = df.drop(columns=["Customer ID", "Customer Name", "Purchase Date", "Churn"])
y = df["Churn"]

# 8. Train-Test Split Manual Menggunakan Pandas (.sample)
# Mengambil 80% data untuk train dan 20% untuk test
train_df = X.sample(frac=0.8, random_state=42)
test_df = X.drop(train_df.index)

train_y = y.loc[train_df.index]
test_y = y.loc[test_df.index]

# 9. Penskalaan (Scaling) Manual - Mirip StandardScaler: (x - mean) / std
# PENTING: mean dan std HANYA dihitung dari data train (mencegah data leakage)
numeric_features = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age', 'Returns']

for col in numeric_features:
    mean_val = train_df[col].mean()
    std_val = train_df[col].std()
    
    # Terapkan ke train set
    train_df[col] = (train_df[col] - mean_val) / std_val
    # Terapkan ke test set menggunakan mean dan std dari train set
    test_df[col] = (test_df[col] - mean_val) / std_val


# 10. Pengodean Kategorikal (One-Hot Encoding) Manual Menggunakan pd.get_dummies
categorical_features = ['Product Category', 'Payment Method', 'Gender']

# Ubah kategori menjadi kolom biner menggunakan get_dummies
train_encoded = pd.get_dummies(train_df, columns=categorical_features, drop_first=False)
test_encoded = pd.get_dummies(test_df, columns=categorical_features, drop_first=False)

# Menyamakan struktur kolom antara train dan test (jika ada kategori yang tidak muncul di salah satu set)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

print("Data berhasil dibersihkan dan diproses secara manual tanpa Scikit-Learn!")
print("Ukuran data train:", train_encoded.shape)
print("Ukuran data test:", test_encoded.shape)