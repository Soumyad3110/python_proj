#EXPLORATORY DATA ANALYSIS ON HOUSING DATA

import numpy as n
import pandas as pandas
import matplotlib.pyplot as plt
import seaborn as sns 


# Read the CSV file into a numpy array
housing_data = n.genfromtxt('c:/Users/KIIT/AppData/Local/Packages/5319275A.WhatsAppDesktop_cv1g1gvanyjgm/TempState/B6622E4EF1A8D811316FE50FD2975FAF/Housing.csv', delimiter=',', skip_header=1)

# Basic numpy operations
print("Array shape:", housing_data.shape)
print("Array dimensions:", housing_data.ndim)
print("Array mean:", n.mean(housing_data))
print("Array standard deviation:", n.std(housing_data))
print("Array minimum:", n.min(housing_data))
print("Array maximum:", n.max(housing_data))

correlation = n.corrcoef(housing_data.T)
print("Correlation matrix:\n", correlation)

# Basic array operations
print("Sum of all elements:", n.sum(housing_data))
print("Column-wise mean:", n.mean(housing_data, axis=0))
print("Row-wise mean:", n.mean(housing_data, axis=1))

#Using the Pandas Library

# Read the CSV file into a pandas array
df = pandas.read_csv("c:/Users/KIIT/AppData/Local/Packages/5319275A.WhatsAppDesktop_cv1g1gvanyjgm/TempState/B6622E4EF1A8D811316FE50FD2975FAF/Housing.csv") 
print(df)
print(df[df['bedrooms']==4]) 
print(df[df['bathrooms']==4])
print(df)
print(df.shape)
print(df.describe())
# Head and Tail
print(df.head())
print(df.tail())

# Indexing and Slicing
print(df.columns)

#data types
print(df.dtypes)

# information about the data
print(df.info())

# Null values
print(df.isna().sum())  
print(df.isna().mean()*100)
print(df.isnull())

# Filling the null values  with 0
print(df.fillna(0))

print(df)

# Dropping the null values
print(df.dropna())

#Duplicate data
print(df.duplicated().sum())
print(df[df.duplicated])

print(df.nunique())
print(df['bedrooms'].unique())
print(df['bedrooms'].value_counts())
print(df['bedrooms'].mean())
print(df['bedrooms'].median())
print(df['bedrooms'].mode())
print(df['bedrooms'].std())
print(df['bedrooms'].var())

print(df['bedrooms'].value_counts())
group=df.groupby('bedrooms')
print(group.get_group(4))
print(group.get_group(4).min())
print(group.get_group(4).max())
print(group['bathrooms'].get_group(4).sum().mean())
print(group['bathrooms'].get_group(2).median())
print(group['bathrooms'].get_group(2).mode())
print(group['bathrooms'].get_group(2).std())
print(group['bathrooms'].get_group(2).var())
print(df.set_index('basement'))
print(df.reset_index(inplace=True))

#Using the Matplotlib Library

# Histogram of house prices
plt.figure(figsize=(10,6))
plt.hist(df['price'], bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Box plot of bedroom counts
plt.figure(figsize=(8,6))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('House Prices by Number of Bedrooms')
plt.show()

# Scatter plot of price vs square footage
plt.figure(figsize=(10,6))
plt.scatter(df['area'], df['price'])
plt.title('Price vs Square Footage')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
a = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(a.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Bar plot of average price by bedroom count
plt.figure(figsize=(10,6))
df.groupby('bedrooms')['price'].mean().plot(kind='bar')
plt.title('Average House Price by Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Average Price')
plt.show()

#Using the Seaborn Library

# Distribution plot of prices
sns.set_style("whitegrid")
sns.displot(data=df, x='price', kde=True)
plt.title('Price Distribution')
plt.show()

# Violin plot of prices by bedroom count
plt.figure(figsize=(10,6))
sns.violinplot(x='bedrooms', y='price', data=df)
plt.title('Price Distribution by Bedrooms')
plt.show()

# Joint plot of price vs living space
sns.jointplot(data=df, x='area', y='price', kind='reg')
plt.show()

# Pair plot of main numerical features
sns.pairplot(df[['price', 'bedrooms', 'bathrooms', 'area']])
plt.show()

# Box plot with categorical variables
plt.figure(figsize=(12,6))
sns.boxenplot(x='bedrooms', y='price', hue='waterfront', data=df)
plt.title('Price Distribution by Bedrooms and Waterfront')
plt.show()
