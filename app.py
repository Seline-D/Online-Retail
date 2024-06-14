# Import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Produce consistent results across different runs.
SEED = 123
np.random.seed(SEED)

##Import data
data = pd.read_csv("C:\\Users\\hp\\Desktop\\Seline\\online retail.csv", encoding="latin1")
data.head()

df = pd.DataFrame(data)
print(df)

#check summary statistics of numerial values
df.describe().T

DATA EXPLOITATION

#Check summary statistics of categorical data
df.describe(include='object').T

#check the dataset data information
df.info()

# check sum of null values
df.isnull().sum()

#check shape (ROWS AND COLUMNS)
df.shape


#CREATE A TABLE TO SHOW EACH COLUMN INFO
def stats(col):
    return pd.Series({
        'Data Type': col.dtype,
        'Non-Missing Values': col.notna().sum(),
        'Missing Values':col.isnull().sum(),
        'Unique Values':col.nunique(),
        'No Of Duplicates': col.duplicated().sum()
    })
stats_table = df.apply(stats, axis=0)

stats_table

#Data Processing
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

#Purchase Frequency
purchase_frequency = data.groupby('Customer ID')['Invoice'].nunique()

#Most Purchased Item
most_purchased_item = data['StockCode'].value_counts().idxmax()

#Highest quantities purchased
highest_quantities_purchased = data.groupby('StockCode')['Quantity'].sum().idxmax()

#Purchase per region
purchase_per_region = data.groupby('Country')['Invoice'].nunique()

#Most recent purchase
most_recent_purchase = data['InvoiceDate'].max()

#Customers per item
customers_per_item = df.groupby('StockCode')['Customer ID'].nunique(print(customers_per_item)

#Customer per top 10 items
top_items_customers = customers_per_item.nlargest(10)

#Frequently purchased item
top_10_items = df.groupby('StockCode').size().nlargest(10)
print(top_10_items)

#Display results
print("Frequency of Purchase:\n", purchase_frequency)
print("\nMost Purchased Item:", most_purchased_item)
print("\nHighest Quantities Purchased Item:", highest_quantities_purchased)
print("\nPurchase Per Region:\n", purchase_per_region)
print("\nMost Recent Time Purchased:", most_recent_purchase)
print(top_10_items)
print(customers_per_item)

# Customers for top 10 regions
top_10_regions = purchase_per_region.nlargest(10)

#Data Plotting
plt.figure(figsize=(12, 6))

# Plotting top 10 frequently purchased items
plt.figure(figsize=(10, 6))
top_10_items.plot(kind='bar')
plt.title('Top 10 Frequently Purchased Items')
plt.xlabel('Stock Code')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting top 10 regions
plt.figure(figsize=(10, 6))
top_10_regions.plot(kind='bar', color='skyblue')
plt.title('Top 10 Regions by Purchase Count')
plt.xlabel('Region')
plt.ylabel('Purchase Count')
plt.xticks(rotation=45)
plt.gca().set_xticklabels(top_10_regions.index, rotation=45, ha='right')  # Set xtick labels for top 10 regions
plt.tight_layout()
plt.show()

# Plotting number of customers per item
plt.figure(figsize=(10, 6))
customers_per_item.plot(kind='bar', color='lightblue')
plt.title('Number of Customers per Item')
plt.xlabel('Stock Code')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top_items_customers = customers_per_item.nlargest(10)

# Plotting number of customers per top items
plt.figure(figsize=(10, 6))
top_items_customers.plot(kind='bar', color='lightblue')
plt.title('Number of Customers per Top Items')
plt.xlabel('Stock Code')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
