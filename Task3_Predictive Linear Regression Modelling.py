#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from math import sqrt


# **LOADING THE DATASET**

# In[2]:


house=pd.read_csv("data.csv")


# In[3]:


house.sample(10)


# **CHECKING VARIOUS ATTRIBUTES OF DATASET**

# In[4]:


house.shape


# In[5]:


house.info()


# In[6]:


house.describe()


# **DATA CLEANING**

# In[7]:


#checking percentage of missing values in each attribute
null_col = [x for x in house.columns if house[x].isnull().sum()>=1 ]
for x in null_col:
    print(x,":",np.round(100*house[x].isnull().sum()/len(house),3),"% are missing values")


# There are no null values

# In[8]:


#constructing new feature of total square feet
house['tot_sqft']=house['sqft_living']+house['sqft_lot']+house['sqft_above']+house['sqft_basement']


# In[9]:


house.drop(columns=["sqft_living",'sqft_lot','sqft_above','sqft_basement'],inplace=True)


# In[10]:


#constructing new feature price per square feet
house['price_per_sqft']= house['price']/house['tot_sqft']
house.head()


# In[11]:


house["country"].unique()


# In[12]:


house.drop("country",axis=1,inplace=True)


# In[13]:


house["city"].nunique()


# In[14]:


city_stats= house.groupby('city')['city'].agg('count').sort_values()
city_stats


# In[15]:


city_stats_lessthan_15=city_stats[city_stats<=15]
city_stats_lessthan_15


# In[16]:


#changing city to "others" where city count is less than 15
house.city= house.city.apply(lambda x: 'other' if x in city_stats_lessthan_15 else x)


# In[17]:


house.groupby('yr_renovated')['yr_renovated'].agg('count').sort_values()


# In[18]:


#As 2735 houses out of 4600 have not been renovated, thus we can easily drop this column
house.drop("yr_renovated",axis=1,inplace=True)


# In[19]:


house.drop(columns=["street","statezip"],inplace=True)


# In[20]:


#binning
house["yr_built"].unique()


# In[21]:


bins=range(min(house["yr_built"]),max(house["yr_built"])+2,19)
bins


# In[22]:


bin_names=["{0}-{1}".format(i,i+19) for i in range(min(house["yr_built"]),max(house["yr_built"])+1,20) ]
bin_names


# In[23]:


house["yr_built_bins"]=pd.cut(house.yr_built, bins ,labels=bin_names,right=False)
house["yr_built_bins"].value_counts()


# In[24]:


house.sample(10)


# In[25]:


house.groupby('waterfront')['waterfront'].agg('count')


# In[26]:


house.groupby('view')['view'].agg('count')


# In[27]:


house["bedrooms"]=house["bedrooms"].astype(int)


# In[28]:


house["floors"]=house["floors"].astype(int)


# In[29]:


house["bathrooms"]=house["bathrooms"].astype(int)


# In[30]:


house.drop(columns=["date","condition"],inplace=True)


# In[31]:


obj = (house.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))
 
int_ = (house.dtypes == 'int64')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))
 
fl = (house.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# In[32]:


house.sample(10)


# **DATA VISUALIZATION**

# In[33]:


house.corr()["price"].sort_values()


# In[34]:


annot_kws = {"size": 10}
sns.heatmap(house.corr(),annot=True, annot_kws=annot_kws, fmt=".3f")


# In[35]:


f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(house['price'], color="b");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()


# In[36]:


for i, predictor in enumerate(house.drop(columns=["city","price","tot_sqft","price_per_sqft"])):
    plt.figure(i)
    sns.barplot(data=house, x=predictor, y="price")


# In[37]:


sns.scatterplot(data=house,x='tot_sqft',y='price')


# In[38]:


sns.scatterplot(data=house,x='price_per_sqft',y='price')


# **ROBUST SCALING TO DEAL WITH OUTLIERS**
# 
# 

# In[33]:


from sklearn import preprocessing
robust=preprocessing.RobustScaler()

num_var=house[["price","price_per_sqft","tot_sqft"]]
robust_df = robust.fit_transform(num_var)
robust_df = pd.DataFrame(robust_df, columns =["price","price_per_sqft","tot_sqft"])


# In[34]:


robust_df


# In[35]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# fit the scaler to the train set, it will learn the parameters like mean and standard deviation
scaler.fit(robust_df)
#transforminnumg the data
trans_col=scaler.transform(robust_df)
house = pd.concat([house.drop(columns=["price","price_per_sqft","tot_sqft"]),pd.DataFrame(trans_col,columns=house[["price","price_per_sqft","tot_sqft"]].columns)],axis=1)


# In[36]:


house.sample(10)


# In[37]:


house=pd.get_dummies(house,columns=['city'],drop_first=True)


# **MODELLING**

# In[38]:


house.sample(10)


# In[39]:


from sklearn.model_selection import train_test_split
y = house["price"] 
x = house.drop(columns=["price","yr_built_bins"])


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# In[49]:


def mod(model):
 model.fit(x_train,y_train)
 y_pred = model.predict(x_test)
 r2score = r2_score(y_test, y_pred)
 rms = sqrt(mean_squared_error(y_test,y_pred))
 k_folds = KFold(n_splits = 5)
 scores = cross_val_score(model,x_train,np.ravel(y_train), cv = k_folds)
    
 print("Model :", model)
 print("r2 score : ", r2score)
 print("Root Mean Error : ", rms)
 print("Cross Validation Scores: ", scores)
 print("Average CV Score: ", scores.mean())

 fig = plt.figure()
 ax = fig.add_subplot(111)

 ax.errorbar(y_test, y_pred, fmt='o')
 ax.errorbar([1, y_test.max()], [1, y_test.max()])


# In[50]:


from sklearn.linear_model import LinearRegression
mod(LinearRegression())


# In[ ]:




