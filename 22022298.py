# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from statsmodels.tsa.stattools import adfuller

"""This code imports the statsmodels.tsa.stattools library's adfuller function.
The Augmented Dickey-Fuller unit root test, a widely used statistical test for stationarity
of a time series, is carried out using this function.
"""

# seasonal_decompose() is used for time series decomposition 

from statsmodels.tsa.seasonal import seasonal_decompose

# Import custom module

import cluster_tools as ct

# Set the option to display all columns

pd.set_option('display.max_columns', None)

# Setting the specific columns 

columnsName = ["DATE", "value"]

# Read in the "Electric_Production.csv" file as a pandas dataframe and setting the specific columns

df = pd.read_csv("ImportsCrudeOil.csv",names = columnsName, header = 0, parse_dates = [0])

#storing in the a new variable

newdata = df.to_numpy()

# Transpose of dataset

transpose = newdata.T

#print Transpose

print(transpose)

# Checking top entries

df.head()

# Setting date as index

df = df.set_index(['DATE'])

#Filling null values

df = df.fillna(0)

#Again Verifing the dataset

df.head()

#checking info

df.info()

df.tail()

# Calculate the rolling mean and rolling std according to months

rolling_mean = df.rolling(window=12).mean()
rolling_std = df.rolling(window=12).std()

plt.figure(figsize = (12,8), dpi=300)

# displaying the plot

plt.plot(df, label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')

# Setting the x label and y label

plt.xlabel('Date', size = 12)
plt.ylabel('Import Crude Oil', size  = 12)

# Setting the legend at the upper left position

plt.legend(loc = 'upper left')

# setting the super title and title

plt.suptitle('Rolling Statistics', size = 14)
plt.title("22022298")

# Saving the plot

plt.savefig("RollingOutput.png")

# displaying the plot

plt.show()


# Setting the figuresize

plt.figure(figsize=(12,8))

# displaying the plot

plt.plot(df['value'])

# Setting the x label and y label

plt.xlabel("Dates")
plt.ylabel("Import Crude Oil")

# setting the super title and title

plt.suptitle("Import Crude Oil TimeSeries")
plt.title("22022298")

plt.savefig("Graph.png")

# displaying the plot

plt.show()

# Use the augmented Dickey-Fuller test to check for stationarity

adft = adfuller(df,autolag="AIC")

# Create a DataFrame with ADF test results

output_df = pd.DataFrame({
    "Values": [adft[0], adft[1], adft[2], adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']],
    "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",               
               "critical value (1%)", "critical value (5%)", "critical value (10%)"]
})

# Print the DataFrame

print(output_df)


# Calculate autocorrelation at lag 1

autocorrel_lag1 = df['value'].autocorr(lag=1)

# Print the result

print("One Month Lag: ", autocorrel_lag1)


# Calculate autocorrelation for lags of 3, 6, and 9 months

autocorrel_lag3 = df['value'].autocorr(lag=3)
autocorrel_lag6 = df['value'].autocorr(lag=6)
autocorrel_lag9 = df['value'].autocorr(lag=9)

# getting  the results using print() function

print("Six Month Lag:", autocorrel_lag6)

#seasonal decomposition of time series data to calculate the trend and using the additive model

decompose = seasonal_decompose(df['value'], model='additive', period=7)

# Plot decomposition

decompose.plot()
plt.savefig("Shabir-22022298.png",dpi=300)
plt.show()

# Read the Second dataset from a CSV file 

data = pd.read_csv("forest.csv")

# getting the top rows using head function

data.head()

# Filling the null values and storing in the same dataset to replace the previous one

data = data.fillna(0)


# Selecting columns from dataframe

ClusteringData = data[['1990', '2000', '2010', '2020']]

corr = ClusteringData.corr()

corr

ct.map_corr(ClusteringData)

#Saving the plot and showing plot

plt.savefig("heatmap.png")

plt.show()

# Plot a scatter matrix of ClusteringData

pd.plotting.scatter_matrix(ClusteringData, figsize=(12, 12), s=5, alpha=0.8)

# Save the scatter matrix plot as an image file

plt.savefig("Matrix.png", dpi=300)

# Display the scatter matrix plot
plt.show()

# Selecting '1990' and '2020' columns from 'ClusteringData' DataFrame
req_df = ClusteringData[['1990', '2020']]

# Dropping rows with null values
req_df = req_df.dropna()

# Resetting index
req_df = req_df.reset_index()

# Printing first 15 rows of the DataFrame
print(req_df.iloc[0:15])

# Dropping 'index' column
req_df = req_df.drop('index', axis=1)

# Printing first 15 rows of the DataFrame
print(req_df.iloc[0:15])



# Scale the dataframe
df_norm, df_min, df_max = ct.scaler(req_df)


print()

print('n  value')

for ncluster in range(2, 10):
    # setup the  cluster with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    #fitting the dataset 
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    
    cen = kmeans.cluster_centers_
    
    print(ncluster, skmet.silhouette_score(req_df, labels))


# Set number of clusters
ncluster = 5

# Perform KMeans clustering
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# Extract x and y coordinates of cluster centers
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]


# Create scatter plot with labeled points and cluster centers
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm['1990'], df_norm['2020'], 10, labels, marker='o', cmap=cm)
plt.scatter(xcen, ycen, 45, 'k', marker='d')
plt.suptitle("Five Clusters", size = 20)
plt.title("21082679",size = 18)
plt.xlabel("Forest(1970)", size = 16)
plt.ylabel("Forest(2020)", size = 16)
plt.savefig("Five Clusters.png", dpi=300)
plt.show()

print(cen)
# Applying the backscale function to convert the cluster centre
scen = ct.backscale(cen, df_min, df_max)
print()
print(scen)
xcen = scen[:, 0]
ycen = scen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(req_df["1990"], req_df["2020"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.suptitle("Three Centered Clusters", size = 20)
plt.title("21082679",size = 18)
plt.xlabel("Forest(1970)", size = 16)
plt.ylabel("Forest(2020)", size = 16)
plt.savefig("Five Centered Clusters.png", dpi=300)
plt.show()