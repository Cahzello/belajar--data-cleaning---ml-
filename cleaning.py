import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv('./data/ecommerce_customer_data_large.csv')

# mengapus data yang duplikat
df = df.drop_duplicates()
# mengahpus kolom umur karena ada 2 kolom yang merepresentasikan data sama
df = df.drop(columns=["Age"])
# mengganti tipe data purchase date ke datetime dari string
df["Purchase Date"] = pd.to_datetime(df["Purchase Date"])
# mengisi data yang kosong
df['Returns'] = df['Returns'].fillna(0)
# menghitung kembali total purchase karena ada kekeliuran pada data
df["Total Purchase Amount"] = df["Product Price"] * df["Quantity"]

# menghapus outlier menggunakan metode iqr pada kolom numerik
kolom_numerik = ['Product Price', 'Quantity', 'Total Purchase Amount', 'Customer Age', 'Returns']
for kolom in kolom_numerik:
    q1 = df[kolom].quantile(0.25)
    q3 = df[kolom].quantile(0.75)
    iqr = q3-q1
    lower_bound = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    df = df[(df[kolom] >= lower_bound) & (df[kolom] <= upper_bound)]
    


# # print(df.info())
# print(df.describe())
# print(df.head())
print(df)


