import pandas as pd
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython import get_ipython
from pandas.core import datetools
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression


df = pd.read_csv(r"SeaWithMean.csv").set_index('date')

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

"""get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [14, 8]  
df.sealevelpressure_1.hist()  
plt.title('Distribution of maxhumidity_1')  
plt.xlabel('sealevelpressure')  
plt.show() """

df = df.dropna() 
df.corr()[['meantemp']].sort_values('meantemp') 

predictors = ['sealevelpressure_1' , 'sealevelpressure_2' , 'sealevelpressure_3', 'sealevelpressure_4',
              'windspeed_1' , 'windspeed_2' , 'windspeed_3','windspeed_4', 
              'dewpoint_1' , 'dewpoint_2' , 'dewpoint_3', 'dewpoint_4',
              'meantemp_1',  'meantemp_2',  'meantemp_3', 'meantemp_4',
              ] 
df2 = df[['meantemp'] + predictors]

get_ipython().run_line_magic('matplotlib', 'inline')

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [16, 22]

# call subplots specifying the grid structure we desire and that 
# the y axes should be shared
fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True)

# Since it would be nice to loop through the features in to build this plot
# let us rearrange our data into a 2D array of 6 rows and 3 columns
arr = np.array(predictors).reshape(3, 3)

# use enumerate to loop over the arr 2D array of rows and columns
# and create scatter plots of each meantempm vs each feature
for row, col_arr in enumerate(arr):  
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2['airtemperature'])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel='airtemperature')
        else:
            axes[row, col].set(xlabel=feature)
plt.show() 

import statsmodels.api as sm

# separate our my predictor variables (X) from my outcome variable y
X = df2[predictors]  
y = df2['meantemp']

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)  
X.iloc[:5, :5]

alpha = 0.05

# (2) Fit the model
model = sm.OLS(y, X).fit()

# (3) evaluate the coefficients' p-values
model.summary()

import operator
q = True
while (q):
    max_index, max_value = max(enumerate(model.pvalues), key=operator.itemgetter(1))
    if (max_value > 0.05):
        X = X.drop(model.pvalues.index[max_index], axis=1)
        model = sm.OLS(y, X).fit()
        model.summary()
    else:
        q = False

"""X = X.drop('sealevelpressure_3', axis=1)

# (5) Fit the model 
model = sm.OLS(y, X).fit()

model.summary() """

 
X = X.drop('const', axis=1)

X_train = X.head(800)
X_test = X.tail(426)

y_train = y.head(800)
y_test = y.tail(426)



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12) 

regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)
from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))  
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))  
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))



plt.plot(X_test.index[50:80], y_test[50:80],'-')
plt.plot(X_test.index[50:80], prediction[50:80],'-')

plt.plot(X_test.index[0:100],prediction[0:100],X_test.index[0:100], y_test[0:100],'-')




