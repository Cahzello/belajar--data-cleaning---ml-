import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv('./data/ecommerce_customer_data_large.csv')

# ==========================================
# TAHAP PEMBERSIHAN DATA
# ==========================================

# 1. menghapus data yang duplikat
df = df.drop_duplicates()
# 2. menghapus kolom umur karena ada 2 kolom yang merepresentasikan data sama
df = df.drop(columns=["Age"])
# 3. mengganti tipe data purchase date ke datetime dari string
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"])
# 4. mengisi data yang kosong pada kolom Returns
df['Returns'] = df['Returns'].fillna(0)
# 5. menghitung kembali total purchase karena ada kekeliruan pada data
df["Total Purchase Amount"] = df["Product Price"] * df["Quantity"]


# ==========================================
# TAHAP PRA-PEMROSESAN MANUAL (TANPA SKLEARN)
# ==========================================

# 6. menghapus kolom identitas yang tidak relevan untuk pemodelan
df = df.drop(columns=["Customer ID", "Customer Name", "Purchase Date"])

# 7. memisahkan Fitur (X) dan Target (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 8. Train-Test Split Manual Menggunakan Pandas (.sample)
# Mengambil 80% data untuk train dan 20% untuk test
train_df = X.sample(frac=0.8, random_state=42).copy()
test_df = X.drop(train_df.index).copy()

train_y = y.loc[train_df.index]
test_y = y.loc[test_df.index]

# 9. Menghapus outlier menggunakan metode IQR
# PENTING: batas IQR dihitung HANYA dari data train (mencegah data leakage),
# lalu batas yang sama diterapkan ke data test.
# Kolom Returns (biner 0/1) tidak diikutkan karena kuartilnya tidak bermakna.
numeric_features = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age']

for col in numeric_features:
    q1 = train_df[col].quantile(0.25)
    q3 = train_df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    train_df = train_df[(train_df[col] >= lower_bound) & (train_df[col] <= upper_bound)]
    test_df = test_df[(test_df[col] >= lower_bound) & (test_df[col] <= upper_bound)]

# selaraskan target setelah baris outlier dihapus
train_y = train_y.loc[train_df.index]
test_y = test_y.loc[test_df.index]

# 10. Penskalaan (Scaling) Manual - Mirip StandardScaler: (x - mean) / std
# PENTING: mean dan std HANYA dihitung dari data train (mencegah data leakage)
for col in numeric_features:
    mean_val = train_df[col].mean()
    std_val = train_df[col].std()
    # Terapkan ke train set
    train_df[col] = (train_df[col] - mean_val) / std_val
    # Terapkan ke test set menggunakan mean dan std dari train set
    test_df[col] = (test_df[col] - mean_val) / std_val

# 11. Pengodean Kategorikal (One-Hot Encoding) Manual Menggunakan pd.get_dummies
# drop_first=True untuk menghindari jebakan dummy variable (dummy variable trap)
categorical_features = ['Product Category', 'Payment Method', 'Gender']

train_encoded = pd.get_dummies(train_df, columns=categorical_features, drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)

# Menyamakan struktur kolom antara train dan test
# (jika ada kategori yang tidak muncul di salah satu set)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

# ==========================================
# RINGKASAN HASIL
# ==========================================

print("Data berhasil dibersihkan dan diproses secara manual tanpa Scikit-Learn!")
print("Jumlah baris sebelum hapus outlier:", len(df))
print("Jumlah baris setelah hapus outlier:", len(train_encoded) + len(test_encoded))
print("Distribusi Churn:\n", y.value_counts(normalize=True).rename("proporsi"))
print("Ukuran data train:", train_encoded.shape)
print("Ukuran data test:", test_encoded.shape)
print("\nContoh data train yang sudah diproses:")
print(train_encoded.head())
