#!/usr/bin/env python
# coding: utf-8

# 

# Broadly listing down the steps to be followed:
# - Loading data, Understading the data & Data cleaning 
# - Outlier Analysis and EDA
# - Prepare the data for modelling
# - Model building using both K-Means and Hierarchical modelling
# - Model Analysis
# - Perform Visualizations on the clusters formed
# - Report back at least 5 countries which are in direst need of aid.

# ### 1. Loading data, Understading the data & Data cleaning

# In[1]:


#importing necessary libraries

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Read the data
countries=pd.read_csv(r"C:\Users\vishnu sharma\Downloads\Country-data.csv")


# In[3]:


countries.head()


# In[4]:


#Checking the shape of the dataframe
countries.shape


# In[5]:


#Checking the datatypes of each column and also if there are any missing values
countries.info()


# In[6]:


countries.isnull().sum()


# ###### Based on the above information, our dataframe has no missing values, hence, data cleaning is not required.

# In[7]:


#Understanding the spread of the data
countries.describe()


# ### 2. EDA 

# In[8]:


#Creating distplots of all the numeric vairables to understand the distribution of the data

col=countries.columns
num_col=col[1:]

for i in num_col:
    sns.distplot(countries[i])
    plt.show()


# ### Insights of distplots:
# __child_mort:__ It can be observed that the data is not normally distributed and is biased towards the left. A huge range of countries lies within the bracket of 0-25. child_mort will thus be a helpful variable in cluster profiling.
# 
# __exports:__ The data follows kind of a normal distribution but has a long tail in the end. 
# 
# __health:__ The data though not perfertly normal, follows kind of a normal distribution. 
# 
# __imports:__ The data follows kind of a normal distribution but has a long tail in the end. 
# 
# __income:__ It can be observed that the data is not normally distributed and is biased towards the left. A huge range of countries lies within the bracket of 0-10000. "income" will thus be a helpful variable in cluster profiling.
# 
# __inflation:__ The data follows kind of a normal distribution but has a long tail in the end. 
# 
# __life_expec:__ The data is not normally distributed and is biased towards the right. However, it has a long tail to the left, depicting countries having very low life-expec, which intutively looks like the case with very under-developed countries. 
# 
# __total_fer:__ The data is not normally distributed and has 2 internal clusters around 1-3 and around 5. "total_fer" can also help in cluster profiling. 
# 
# __gdpp:__ It can be observed that the data is not normally distributed and is biased towards the left. A huge range of countries lies within the bracket of 0-10000. "gdpp" will thus be a helpful variable in cluster profiling.
# 

# In[9]:


sns.pairplot(data=countries)
plt.show()


# ### Interesting insights from the above group of scatter plots:
# __income Vs child_mort:__ It can be observed that countries with very low income have really high child_mort rate. child_mort suddenly drops with increase in the income.
# 
# __life_expec Vs child_mort:__ This graph has a linear relation with negative slope, as naturally, life expectancy of a country reduces with increase in child_mort. 
# 
# __income Vs life_expec:__ Naturally the life expectancy of countries increses with increase in the income. However, there is 1 country in the bottom left of the graph, with extremly low life_expec and income. Also there is another country on the top right of the graph, with high income and high life_expec. This clearly shows the extreme difference between a under-developed and developed country.
# 
# __total_fer Vs child_mort:__ This graph has a linear relationship. Signifying the increse in the child_mort cases with increase in fertility. 
# 
# __total_fer Vs income:__ This graph follows a strange pattern. The fertility is really high in countries with low individual income and suddenly drops for countries with high income. This can mean 2 things:
# > 1. Countries with low income might have low literacy rate too, resulting in less awareness towards family planning and birth countrol.
# > 2. Countries with high individual income might see a drop in fertility rate due to high work related stress. 
# 
# __gdpp Vs child_mort:__ This graph clearly shows child_mort to be very low for countries with high gdpp and extremly high for countries with low gdpp. 
# 
# __gdpp Vs income:__ This graph shows a linear relationship between income of an individual and gdpp of the country.
# 
# __gdpp Vs inflation:__ The graph shows that countries with high gdpp has a verylow rate of inflation. Signifying, inflation can kill the gdpp of a country.
# 
# __gdpp Vs life_expec:__ Just like income vs life_expec, a sudden surge is observed in the life expectancy of a county with an increase in gdpp. 

# In[10]:


# Building a heat map to understand co-relations between variables
plt.figure(figsize=(10,8))
sns.heatmap(countries.corr(), annot=True,cmap="Blues")
plt.show()


# ### Insights from the above heat map
# 
# child_mort vs life_expec, income vs gdpp, total_fer vs child_mort and imports vs exports have a very high positive co-relation, which is natural and self-explanatory. 
# 
# While a strong negative co-relation between life_expec & child mort is self explanatory, there is also a strong negative co-relation between total_fer & life_expec. This can be understood from the scatter plots above, where we observed that countries with low income had low life_expec but also very high fertility rate. 

# ## 3.Outlier Analysis & Treatment

# As our objective is to find the countries in direst need of funds, treating outliers in certain variables where the outlier values correspond to poor countries will skew our data due to loss of essential information. Hence, we will not treat such outlier values.(Ex: Outlier towards the higher end in variables like "child_mort"). However, if outlier values of any variable correspond to the richer countries, we will go ahead with capping the outlier values within the IQR. (Ex: Outliers towards the higher end in variables like "income","gdpp") 

# In[11]:


#plotting boxplots to visualize outliers in our data.
sns.boxplot(countries.child_mort)
plt.show()


# In[12]:


countries[(countries.child_mort>=145)]


# As per the approach explained above, not treating outlier values in "child_mort"

# In[13]:


sns.boxplot(countries.exports)
plt.show()


# In[14]:


countries[(countries.exports>=85)]


# Looking at the outlier values and the corresponding countries and their stats, treating the outlier values as per the approach explained above.

# In[15]:


q=countries.exports.quantile(0.97)


# In[16]:


countries["exports"][countries["exports"]>=q] = q


# In[17]:


#checking if outliers are treated successfully
sns.boxplot(countries.exports)
plt.show()


# In[18]:


sns.boxplot(countries.health)
plt.show()


# In[19]:


countries[(countries.health>=14)]


# Looking at the outlier values and the corresponding countries and their stats, treating the outlier values as per the approach explained above.

# In[20]:


q=countries["health"].quantile(0.99)
q


# In[21]:


countries["health"][countries["health"]>=q]=q


# In[22]:


#checking if outliers are treated successfully
sns.boxplot(countries.health)
plt.show()


# In[23]:


sns.boxplot(countries.imports)
plt.show()


# In[24]:


countries[(countries.imports>=100)]


# Looking at the outlier values and the corresponding countries and their stats, treating the outlier values as per the approach explained above.

# In[25]:


q=countries["imports"].quantile(0.97)
q


# In[26]:


countries["imports"][countries["imports"]>=q]=q


# In[27]:


#checking if outliers are treated successfully
sns.boxplot(countries.imports)
plt.show()


# In[28]:


sns.boxplot(countries.income)
plt.show()


# In[29]:


countries[(countries.income>=50000)]


# Looking at the outlier values and the corresponding countries and their stats, treating the outlier values as per the approach explained above.

# In[30]:


q=countries["income"].quantile(0.95)
q


# In[31]:


countries["income"][countries["income"]>=q]=q


# In[32]:


#checking if outliers are treated successfully
sns.boxplot(countries.income)
plt.show()


# In[33]:


sns.boxplot(countries.inflation)
plt.show()


# In[34]:


countries[(countries.inflation>=25)]


# As per the approach explained above, not treating outlier values in "inflation"

# In[35]:


sns.boxplot(countries.life_expec)
plt.show()


# In[36]:


countries[countries.life_expec<50]


# As per the approach explained above, not treating outlier values in "life_expec"

# In[37]:


sns.boxplot(countries.total_fer)
plt.show()


# In[38]:


countries[countries.total_fer>7]


# As per the approach explained above, not treating outlier values in "total_fer"

# In[39]:


sns.boxplot(countries.gdpp)
plt.show()


# Looking at the outlier values and the corresponding countries and their stats, treating the outlier values as per the approach explained above.

# In[40]:


q=countries["gdpp"].quantile(0.85)
q


# In[41]:


countries["gdpp"][countries["gdpp"]>=q]=q


# In[42]:


#checking if outliers are treated successfully
sns.boxplot(countries.gdpp)
plt.show()


# # 4. Prepare the data for modelling

# Converting "exports","health" and "imports" to absolute values, as they are given in the form of percentage of gdpp

# In[43]:


countries.head()


# In[44]:


countries.exports=((countries.exports*countries.gdpp)/100)


# In[45]:


countries.health=((countries.health*countries.gdpp)/100)


# In[46]:


countries.imports=((countries.imports*countries.gdpp)/100)


# In[47]:


countries.head()


# Before scaling the data, checking the Hopkins statistic for our dataframe to vefiry how suitable is our data for clustering.

# In[48]:


from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


# In[49]:


countries_num=countries.drop("country",axis=1)


# In[50]:


hopkins(countries_num)


# Based on the above Hopkins statistic, our data is good to go for clustering exercise.

# #### Rescaling the data using Standard Scaler

# In[51]:


import sklearn
from sklearn.preprocessing import StandardScaler


# In[52]:


scaler=StandardScaler()


# In[53]:


countries_num_scaled=scaler.fit_transform(countries_num)


# In[54]:


countries_num_scaled


# In[55]:


countries_num.head()


# In[56]:


#Converting the numpy array to pandas dataframe
countries_num_scaled=pd.DataFrame(countries_num_scaled)
countries_num_scaled.columns=["child_mort","exports","health","imports","income","inflation","life_expec","total_fer","gdpp"]


# In[57]:


countries_num_scaled.head()


# In[58]:


# Just out of curiosity, verifying Hopkins statistic again to see the effect of scaling on the value of Hopkins statistic.
# It can be observed that scaling did reduce the Hopkis statistic, but our data is still good to go for clustering.
hopkins(countries_num_scaled)


# In[59]:


countries_num_scaled.shape


# # 5. Model building using both K-Means and Hierarchical clustering
# 

# ## K-Means

# In[60]:


from sklearn.cluster import KMeans


# In[61]:


kmeans=KMeans(random_state=100)


# In[62]:


kmeans.fit(countries_num_scaled)


# In[63]:


kmeans.labels_


# #### Choosing the optimal number of clusters using Elbow curve

# In[64]:


range_clusters=[1,2,3,4,5,6]
ssd=[]
for num_clusters in range_clusters:
    kmeans=KMeans(n_clusters=num_clusters,random_state=100)
    kmeans.fit(countries_num_scaled)
    
    ssd.append(kmeans.inertia_)

ssd


# In[65]:


plt.plot(range(1,7),ssd)
plt.show()


# #### Using Silhouette Analysis to explore the optimal number of clusters

# In[66]:


from sklearn.metrics import silhouette_score


# In[67]:


range_clusters=[2,3,4,5,6,7,8,9]

for num_clusters in range_clusters:
    kmeans=KMeans(n_clusters=num_clusters,random_state=100)
    kmeans.fit(countries_num_scaled)
    
    cluster_lables=kmeans.labels_
    silhouette_avg=silhouette_score(countries_num_scaled, cluster_lables)
    print("For num_clusters {0}, silhouette score is {1}".format(num_clusters,silhouette_avg))


# As per the elbow curve, 2 is the optimum number of clusters. But we see a significant drop in SSD with the addition of 3rd cluster as well. Additionally, though the Silhouette score for 2 clusters is highest, we should go with the next best value, which is for 3 clusters, as 2 clusters is not really a good choice. Intutively speaking, I personally expect 3 clusters of something like "Poor", "Medium" and "Rich" countries coming out of this clustering exercise. Hence going ahead with 3 clusters.

# In[68]:


#Creating the final model with 3 clusters
kmeans=KMeans(n_clusters=3,random_state=100)
kmeans.fit(countries_num_scaled)


# In[69]:


#Assigning the cluster labels to the country variable in our previous dataset
countries["cluster"]=kmeans.labels_


# In[70]:


countries.head()


# # 6. Cluster Analysis

# In[71]:


sns.boxplot(x="cluster",y="child_mort",data=countries)
plt.show()


# In[72]:


sns.boxplot(x="cluster",y="income",data=countries)
plt.show()


# In[73]:


sns.boxplot(x="cluster",y="gdpp",data=countries)
plt.show()


# As per the 3 box plots above, cluster 1 has the lowest "child_mort" rate and the highest range of "income" and "gdpp". Whereas, cluster 2 has the highest "child_mort" rate and the lowest range of "income" and "gdpp". 
# 
# Hence, we will go ahead with the below cluster profiling:<br>
# Cluster 1: Rich/Developed countries<br>
# Cluster 0: Developing countries<br>
# Cluster 2: Poor/under-developed countries (Our area of focus)

# ### Countries in the direst need of aid
# Well, as stated in the problem statement, "gdpp", "child_mort" and "income" are to be used for cluster analysis, using the same variables in the same order to sort the final list of countries present in the cluster of  poor countries, to shortlist the top 5 countries which are in direst need of aid.

# In[74]:


countries[countries.cluster==2].sort_values(by=["gdpp","child_mort","income"],ascending=[True,False,True])


# ## Hierarchical Clustring 

# In[75]:


from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[76]:


#single linkage
mergings=linkage(countries_num_scaled,method='single',metric="euclidean")
dendrogram(mergings)
plt.show()


# In[77]:


#complete linkage
mergings=linkage(countries_num_scaled,method='complete',metric="euclidean")
dendrogram(mergings)
plt.show()


# As per the dendrogram above, we see a very thin cluster (of probably 1 country) on the left and all the other countries are part of another cluster. Hence if we cut the dendrogram between 10-8, we get 3 clusters of which, one will be useless. Hence, cutting the dendrogram between 8-6 to get 4 clusters. 

# In[78]:


cluster_H=cut_tree(mergings,n_clusters=4).reshape(-1,)


# In[79]:


countries["cluster_H"]=cluster_H


# In[80]:


countries.head()


# In[81]:


sns.boxplot(x="cluster_H",y="income",data=countries)
plt.show()


# In[82]:


sns.boxplot(x="cluster_H",y="gdpp",data=countries)
plt.show()


# In[83]:


sns.boxplot(x="cluster_H",y="child_mort",data=countries)
plt.show()


# In[84]:


countries[countries.cluster_H==3]


# As observed in the boxplots above, the thinnest cluster 3 consists of only 1 country. Ignoring that, if we analyze the other 3 clusters, cluster 0 is the one with the highest range of "child_mort" rate and lowest "gdpp" and "income". Hence categorizing cluster 0 as the cluster of poor countries. <br>
# 
# As per the logic used in K-Means clustering, sorting the values of cluster-0 to shortlist the countries in the direst need of aid.

# In[85]:


countries[countries.cluster_H==0].sort_values(by=["gdpp","child_mort","income"],ascending=[True,False,True])


# ## 7. Finalizing List of Countries

# Both K-Means and Hierarchical clustering has given the same top-5 countries which are in the direst need of aid. Hopefully, this shows the accuracy of our model. Hence, below are the recommended 5 countries which we need to focus on:

# ### 1) Burundi
# ### 2) Liberia
# ### 3) Congo, Dem. Rep.
# ### 4) Niger
# ### 5) Sierra Leone
# ##### The below list can be given second priority:
# ##### 6) Madagascar
# ##### 7) Mozambique
# ##### 8) Central African Republic
# 

# In[ ]:




