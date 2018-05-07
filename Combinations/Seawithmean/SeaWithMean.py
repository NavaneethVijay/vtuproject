import pandas as pd
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython import get_ipython
from pandas.core import datetools
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 
import statsmodels.api as sm 
import operator
from sklearn.metrics import mean_absolute_error, median_absolute_error 


df = pd.read_csv(r"SeaWithMean.csv")

#df = df.apply(lambda x: x.str.strip()).replace('', np.nan)

features = ['sealevelpressure','airtemperature','dewpoint','winddirection','windspeed', 'meantemp']

N = 1

to_keep = ['sealevelpressure','airtemperature','dewpoint','winddirection','windspeed', 'meantemp']

df = df[to_keep]

df = df.apply(pd.to_numeric, errors='coerce')

mean = df.mean()

df['sealevelpressure'].replace(to_replace=np.nan, value=mean[0])

df['airtemperature'].replace(to_replace=np.nan, value=mean[1])

df['dewpoint'].replace(to_replace=np.nan, value=mean[2])

df['winddirection'].replace(to_replace=np.nan, value=mean[3])

df['windspeed'].replace(to_replace=np.nan, value=mean[4])

df['meantemp'].replace(to_replace=np.nan, value=mean[4])

def derive_nth_day_feature(df, feature, N):  
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements


for feature in features:  
    if feature != 'date':
        for N in range(1, 5):
            derive_nth_day_feature(df, feature, N)


spread = df.describe().T

IQR = spread['75%'] - spread['25%']

spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))

spread.loc[spread.outliers,] 

# data distribution histograms
"""get_ipython().run_line_magic('matplotlib', 'inline')

 for sp in spread.index:
    plt.rcParams['figure.figsize'] = [14, 8]  
    print(sp)
    df[sp].hist() 
    plt.title('Distribution of {}'.format(sp))  
    plt.xlabel(sp)  
    plt.show()"""
# data distribution histograms
    
    
df = df.dropna() 

#pearson correlation
df.corr()[['meantemp']].sort_values('meantemp') 

predictors = ['sealevelpressure_1' , 'sealevelpressure_2' , 'sealevelpressure_3', 'sealevelpressure_4',
              'windspeed_1' , 'windspeed_2' , 'windspeed_3','windspeed_4', 
              'dewpoint_1' , 'dewpoint_2' , 'dewpoint_3', 'dewpoint_4',
              'meantemp_1',  'meantemp_2',  'meantemp_3', 'meantemp_4',
              ] 
df2 = df[['meantemp'] + predictors]




# separate our my predictor variables (X) from my outcome variable y
X = df2[predictors]
y = df2['meantemp']

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)  
X.iloc[:5, :5]

#backwarfd elimintaion
# (1) set a significance value
alpha = 0.05

# (2) Fit the model
model = sm.OLS(y, X).fit()

# (3) evaluate the coefficients' p-values
model.summary()


q = True
while (q):
    max_index, max_value = max(enumerate(model.pvalues), key=operator.itemgetter(1))
    if (max_value > 0.05):
        X = X.drop(model.pvalues.index[max_index], axis=1)
        model = sm.OLS(y, X).fit()
        print(model.summary())
    else:
        q = False
#backwarfd elimintaion
        
        
#training and testing
X_train = X.head(800)
X_test = X.tail(353)

y_train = y.head(800)
y_test = y.tail(353)



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12) 

regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))  
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))  
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))



fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(111)
ax1.plot(y_test.tolist(), label='actual values')
ax1.plot(prediction,label='predicted values')
plt.xlabel('Training dataset')  
plt.ylabel('Mean temperature') 
plt.legend(loc='upper left');
plt.show()








