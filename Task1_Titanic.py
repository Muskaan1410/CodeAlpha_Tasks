#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
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


titanic=pd.read_csv("Titanic-Dataset.csv")


# In[3]:


titanic.sample(10)


# **CHECKING VARIOUS ATTRIBUTES OF DATASET**

# In[4]:


titanic.shape


# In[5]:


titanic.info()


# In[6]:


titanic.describe()


# In[7]:


# percentage of people who survived 
100*titanic["Survived"].value_counts()/len(titanic["Survived"])


# It is clear that data is highly imbalanced.

# **HANDLING MISSING VALUES**

# In[8]:


#checking percentage of missing values in each attribute
null_col = [x for x in titanic.columns if titanic[x].isnull().sum()>=1 ]
for x in null_col:
    print(x,":",np.round(100*titanic[x].isnull().sum()/len(titanic),3),"% are missing values")


# In[9]:


#As 77% data is missing in "Cabin", we can drop this column.
titanic.drop(columns='Cabin', axis=1,inplace=True)
titanic.shape


# In[10]:


#We will replace missing values in "Age" with the mean
titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)


# In[11]:


#We will replace the missing values in "Embarked" with the most frquent value in the column
titanic["Embarked"].fillna(titanic['Embarked'].mode()[0],inplace=True)


# **BINNING**
# 
# 

# In[12]:


titanic["Age"]=titanic["Age"].astype(int)


# In[13]:


bins=range(min(titanic["Age"]),max(titanic["Age"])+2,9)
bins


# In[14]:


bin_names=["{0}-{1}".format(i,i+8) for i in range(min(titanic["Age"]),max(titanic["Age"])+1,9) ]
bin_names


# In[15]:


titanic["age_bins"]=pd.cut(titanic.Age, bins ,labels=bin_names,right=False)
titanic["age_bins"].value_counts()


# **ENCODING CATEGORICAL DATA**
# 

# In[16]:


titanic.info()


# In[17]:


#The only categorical columns that needs to be encoded are "Sex" and "Embarked"


# In[18]:


titanic.replace({'Sex':{'male':0,'female':1}}, inplace=True)


# In[19]:


titanic.replace({'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[20]:


titanic.sample(5)


# **STANDARADIZATION OF FARE COLUMN**
# 
# 

# In[21]:


scaler = StandardScaler()

# fit the scaler to the train set, it will learn the parameters like mean and standard deviation
scaler.fit(titanic[["Fare"]])
#transforming the data
trans_fare=scaler.transform(titanic[["Fare"]])
#concating
titanic = pd.concat([titanic.drop("Fare",axis=1),pd.DataFrame(trans_fare,columns=titanic[["Fare"]].columns)],axis=1)


# In[22]:


np.round(titanic["Fare"].describe(),3)


# **DATA VISUALIZATION**
# 
# 

# In[23]:


for i, predictor in enumerate(titanic.drop(columns=['Survived', 'Ticket', "Fare","Age","Name","PassengerId"])):
    plt.figure(i)
    sns.countplot(data=titanic, x=predictor, hue="Survived")


# **OBSERVATIONS :-**
# 
# 1. The majority of preople who died, belonged to PClass 3.
# Class 1 passengers have a higher survival chance compared to classes 2 and 3. It implies that Pclass contributes a lot to a passenger’s survival rate.
# 
# 2.It can be approximated that the survival rate of men is around 20% and that of women is around 75%. 
# Therefore, whether a passenger is a male or a female plays an important role in determining if one is going to survive.
# 
# 3. The survival and death rate is maximum for age group 27-35.We will plot this graph with the feature "Sex" to get better insights.
# 

# In[24]:


sns.violinplot(data=titanic, x="Sex", y="Age", hue="Survived",split=True)


# **OBSERVATIONS**
# 
# 1.This graph gives a summary of the age range of men, women and children who were saved.
# 
# 2. The survival rate is better for women in the age group 20-40.
# 
# 3. The death rate for peaked in the age group 20-40

# In[25]:


sns.barplot(x ='Survived', y ='Fare', data = titanic)


# **OBSERVATIONS:** 
# 
# It can be concluded that if a passenger paid a higher fare, the survival rate is more.

# In[26]:


titanic['FamilySize'] = titanic['Parch']+titanic['SibSp']


# In[27]:


sns.barplot(x ='FamilySize', y='Survived', data = titanic)


# In[28]:


sns.countplot(x ='FamilySize', hue='Survived', data = titanic)


# **OBSERVATIONS:-**
# 
# 1. Most people travelled alone. If a passenger is alone, the death rate is more.
# 
# 2.If the family size is greater than 4, chances of survival decrease considerably

# In[29]:


titanic.corr()["Survived"].sort_values()


# In[30]:


annot_kws = {"size": 10}
sns.heatmap(titanic.corr(),annot=True, annot_kws=annot_kws, fmt=".3f")


# **OBSERVATIONS:**
# 
# The columns that can be dropped are: 
# PassengerId, Name, Ticket as they are strings, cannot be categorized and don’t contribute much to the outcome. 
# Age: Instead, the age_bins column is retained.

# In[31]:


titanic.drop(columns=["PassengerId","Name","Ticket"], axis=1,inplace=True)


# In[32]:


titanic.head()


# **MODEL BUILDING**

# **WITHOUT PRINCIPAL COMPONENT ANALYSIS**

# In[33]:


y = titanic["Survived"] 
x = titanic.drop(["Survived","age_bins"],axis=1)


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# In[35]:


def mod(model):
 model.fit(x_train,y_train)
 y_pred = model.predict(x_test)
 acc_score = accuracy_score(y_test,y_pred)
 r2score = r2_score(y_test, y_pred)
 prec_score = precision_score(y_test, y_pred)
 rms = sqrt(mean_squared_error(y_test,y_pred))
 k_folds = KFold(n_splits = 5)
 scores = cross_val_score(model,x_train,np.ravel(y_train), cv = k_folds)
    
 print("Model :", model)
 print("The accuracy score : ", acc_score)
 print("r2 score : ", r2score)
 print("Precision score : ", prec_score)
 print("Root Mean Error : ", rms)
 print("Cross Validation Scores: ", scores)
 print("Average CV Score: ", scores.mean())


# In[36]:


mod(LogisticRegression())


# In[37]:


mod(DecisionTreeClassifier())


# In[38]:


mod(RandomForestClassifier())


# In[39]:


mod(SVC())


# In[40]:


acc = []

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train,y_train)
    yhat = knn.predict(x_test)
    acc.append(accuracy_score(y_test,yhat))


plt.figure(figsize=(8,6))
plt.plot(range(1,20),acc, marker = "o")
plt.xlabel("Value of k")
plt.ylabel("Accuracy Score")
plt.title("Find the right k")
plt.xticks(range(1,20))
plt.show()


# In[53]:


mod(KNeighborsClassifier(n_neighbors =5))


# **USING PCA**

# In[42]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(x)
x_pca = pca.transform(x)
 
# Create the dataframe
df_pca = pd.DataFrame(x_pca, columns=['PC1','PC2','PC3'])
print(df_pca)


# In[43]:


x1=df_pca
y1=y


# In[44]:


x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size = 0.2, random_state=1)


# In[45]:


def mod_pca(model):
 model.fit(x1_train,y1_train)
 y1_pred = model.predict(x1_test)
 r2score = r2_score(y1_test, y1_pred)
 acc_score = accuracy_score(y1_test,y1_pred)
 prec_score = precision_score(y1_test, y1_pred, zero_division = 1)
 rms = sqrt(mean_squared_error(y1_test,y1_pred))
 k_folds = KFold(n_splits = 5)
 scores = cross_val_score(model,x1_train,np.ravel(y1_train), cv = k_folds)
    
 print("Model :", model)
 print("The accuracy score : ", acc_score)
 print("r2 score : ", r2score)
 print("Precision score : ", prec_score)
 print("Root Mean Error : ", rms)
 print("Cross Validation Scores: ", scores)
 print("Average CV Score: ", scores.mean())


# In[46]:


mod_pca(LogisticRegression())


# In[47]:


mod_pca(DecisionTreeClassifier())


# In[48]:


mod_pca(RandomForestClassifier())


# In[49]:


mod_pca(SVC())


# In[50]:


acc = []

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x1_train,y1_train)
    yhat = knn.predict(x1_test)
    acc.append(accuracy_score(y1_test,yhat))


plt.figure(figsize=(8,6))
plt.plot(range(1,20),acc, marker = "o")
plt.xlabel("Value of k")
plt.ylabel("Accuracy Score")
plt.title("Find the right k")
plt.xticks(range(1,20))
plt.show()


# In[51]:


mod_pca(KNeighborsClassifier(n_neighbors =3))


# In[ ]:




