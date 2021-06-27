#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#to split data for train and test
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
#(RANdom SAmple Consensus) algorithm.~rastgele örneklerin benzerliği üstüne-robust
from sklearn.linear_model import TheilSenRegressor
#Theil-Sen Estimator: robust multivariate regression model-çok değişkenli regresyon modeli
#from sklearn.linear_model import SGDRegressor
#SGD's results are terrible. So commented them.
import matplotlib.pyplot as plt
#visualization
import seaborn as sns
#another visiualization
from sklearn import metrics
#to see memory usage


# In[2]:


#couldnt get it done in loop so did it one by one
#i=0
#while i<10:
#    data[i]=arff.loadarff([i]+'.arff')
#    df[i]=pd.DataFrame(data[i][0])
#    i+=1

data0=arff.loadarff('0.arff')
df0=pd.DataFrame(data0[0])
data1=arff.loadarff('1.arff')
df1=pd.DataFrame(data1[0])
data2=arff.loadarff('2.arff')
df2=pd.DataFrame(data2[0])
data3=arff.loadarff('3.arff')
df3=pd.DataFrame(data3[0])
data4=arff.loadarff('4.arff')
df4=pd.DataFrame(data4[0])
data5=arff.loadarff('5.arff')
df5=pd.DataFrame(data5[0])
data6=arff.loadarff('6.arff')
df6=pd.DataFrame(data6[0])
data7=arff.loadarff('7.arff')
df7=pd.DataFrame(data7[0])
data8=arff.loadarff('8.arff')
df8=pd.DataFrame(data8[0])
data9=arff.loadarff('9.arff')
df9=pd.DataFrame(data9[0])


# In[3]:


frames=[df1, df2, df3, df4, df5, df6, df7, df8, df9] #geçici dfleri list halinde frame'e atıyoruz
df=pd.concat(frames, ignore_index=True) #bu framelerden son olarak bütün dataları içeren df'i oluşturuyoruz


# In[4]:


df


# In[5]:


df['pred_Activity'] = df['pred_Activity'].str.decode('utf-8')
#no error for multilanguage


# In[6]:


df.cosmed.min()


# In[7]:


df.pred_Activity.value_counts()#spesifik kolunda neyden kaç tane var


# In[8]:


dummies = pd.get_dummies(df.pred_Activity, drop_first=True)
#one hot encoding, drop_first true ile gereksiz 1 sütundan kurtuluyoruz


# In[9]:


dummies


# In[10]:


df.drop('pred_Activity',axis=1,inplace=True)
#dismiss the non number values


# In[11]:


df=df.join(dummies)
#use these numeric values instead of those not


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


a = df[["peakCount","Z_breathRate","Z_heartRate","Z_skinTemp","cycling","kneeling","lying","running","sitting","standing","standing_leaning","transition","walking"]]
#taking datas except the one we will get with regression, cosmed.


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(a, df["cosmed"], test_size=0.33)
#we can change test_size due to project requirements, also random_state=(int) can be added for multiwork


# In[16]:


x_train


# In[17]:


x_test


# In[18]:


lm=LinearRegression()
ran=RANSACRegressor()
theil=TheilSenRegressor()
#sgd=SGDRegressor()


# In[19]:


lm.fit(x_train,y_train)


# In[20]:


ran.fit(x_train, y_train)


# In[21]:


theil.fit(x_train, y_train)


# In[22]:


#sgd.fit(x_train, y_train)


# In[23]:


predictlm=lm.predict(x_test)
predictran=ran.predict(x_test)
predicttheil=theil.predict(x_test)
#predictsgd=sgd.predict(x_test)


# In[24]:


plt.scatter(y_test,predictlm)


# In[25]:


plt.scatter(y_test,predictran)


# In[26]:


plt.scatter(y_test,predicttheil)


# In[27]:


#plt.scatter(y_test,predictsgd)


# In[28]:


sns.distplot((y_test-predictlm))
#shows that predict is very good but its not a perf metric


# In[29]:


sns.distplot((y_test-predictran))


# In[30]:


sns.distplot((y_test-predicttheil))


# In[31]:


#sns.distplot((y_test-predictsgd))


# In[32]:


df.corr()
#corr matrix. +1 show that they are parallel datas but might not show causality


# In[33]:


df.memory_usage(deep=True)
#be sure that there is not and wont be any memory problem


# In[34]:


metrics.mean_absolute_error(y_test,predictlm)


# In[35]:


metrics.mean_absolute_error(y_test,predictran)


# In[36]:


metrics.mean_absolute_error(y_test,predicttheil)


# In[37]:


#metrics.mean_absolute_error(y_test,predictsgd)


# In[38]:


df.cosmed.describe()


# In[39]:


metrics.mean_squared_error(y_test, predictlm)


# In[40]:


metrics.mean_squared_error(y_test, predictran)


# In[41]:


metrics.mean_squared_error(y_test, predicttheil)


# In[42]:


np.sqrt(metrics.mean_squared_error(y_test,predictlm))


# In[43]:


np.sqrt(metrics.mean_squared_error(y_test,predictran))


# In[44]:


np.sqrt(metrics.mean_squared_error(y_test,predicttheil))


# In[45]:


metrics.r2_score(y_test, predictlm)


# In[46]:


metrics.r2_score(y_test, predictran)


# In[47]:


metrics.r2_score(y_test, predicttheil)


# In[48]:


#metrics.r2_score(y_test, predictsgd)


# In[49]:


y_test


# In[ ]:




