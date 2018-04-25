
# coding: utf-8

# In[223]:


import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\LalithaShankar\Desktop\FinalYearProject\TrialAndError\SeaST.csv")


# In[226]:


df = df.apply(lambda x: x.str.strip()).replace('', np.nan)


# In[224]:


features = ['Sea_Level_Pressure','Air_Temperature','Dew_Point_Temperature','Wind_Direction','Wind_Speed']


# In[225]:


N = 1


# In[227]:


to_keep = ['Sea_Level_Pressure','Air_Temperature','Dew_Point_Temperature','Wind_Direction','Wind_Speed']


# In[228]:


to_keep


# In[229]:


df = df[to_keep]


# In[230]:


df.columns


# In[231]:


df = df.apply(pd.to_numeric, errors='coerce')


# In[232]:


mean = df.mean()


# In[233]:


mean


# In[234]:


df['Sea_Level_Pressure'].replace(to_replace=np.nan, value=mean[0])


# In[235]:


df['Dew_Point_Temperature'].replace(to_replace=np.nan, value=mean[1])


# In[236]:


df['Air_Temperature'].replace(to_replace=np.nan, value=mean[2])


# In[237]:


df['Wind_Direction'].replace(to_replace=np.nan, value=mean[3])


# In[238]:


df['Wind_Speed'].replace(to_replace=np.nan, value=mean[4])


# In[239]:


def derive_nth_day_feature(df, feature, N):  
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements


# In[240]:


for feature in features:  
    if feature != 'Date_of_observation':
        for N in range(1, 4):
            derive_nth_day_feature(df, feature, N)


# In[241]:


df.columns


# In[242]:


df.info() 


# In[243]:


df


# In[244]:


df.describe().T


# In[245]:


spread = df.describe().T



# In[246]:


IQR = spread['75%'] - spread['25%']


# In[247]:


spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))


# In[248]:


spread.loc[spread.outliers,] 


# In[249]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [14, 8]  
df.Sea_Level_Pressure_1.hist()  
plt.title('Distribution of maxhumidity_1')  
plt.xlabel('Sea_Level_Pressure_1')  
plt.show() 

