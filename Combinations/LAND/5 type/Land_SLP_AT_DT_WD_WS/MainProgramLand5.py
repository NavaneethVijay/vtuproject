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

df = pd.read_csv(r"Land_SLP_AT_DT_WD_WS.csv").set_index('date')


#tmp = df[['meantempm', 'meandewptm']].head(10)  

N = 1
features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",  
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm", 'sealevelpressure',	
            'airtemperature',	'dewpoint' ,'winddirection', 'windspeed'
]

def derive_nth_day_feature(df, feature, N):  
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

for feature in features:  
    if feature != 'date':
        for N in range(1, 5):
            derive_nth_day_feature(df, feature, N)
#df.columns 
#df.shape

#Data_Cleaning 
# make list of original features without meantempm, mintempm, and maxtempm
to_remove = [feature  
             for feature in features 
             if feature not in ['meantempm', 'mintempm', 'maxtempm']]

# make a list of columns to keep
to_keep = [col for col in df.columns if col not in to_remove]

# select only the columns in to_keep and assign to df
df = df[to_keep]  
#df.columns 
#df.info() 
df = df.apply(pd.to_numeric, errors='coerce')  
#df.info()

# Call describe on df and transpose it due to the large number of columns
spread = df.describe().T

# precalculate interquartile range for ease of use in next calculation
IQR = spread['75%'] - spread['25%']

# create an outliers column which is either 3 IQRs below the first quartile or
# 3 IQRs above the third quartile
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))

# just display the features containing extreme outliers
spread.loc[spread.outliers,]   

get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = [14, 8]  
df.maxhumidity_1.hist()  
plt.title('Distribution of maxhumidity_1')  
plt.xlabel('maxhumidity_4')  
plt.show() 

df.minpressurem_1.hist()  
plt.title('Distribution of minpressurem_1')  
plt.xlabel('minpressurem_1')  
plt.show() 

for precip_col in ['precipm_1', 'precipm_2', 'precipm_3']:  
    # create a boolean array of values representing nans
    missing_vals = pd.isnull(df[precip_col])
    df[precip_col][missing_vals] = 0
    #re,moving all the NaN values
df = df.dropna()

#finding corelleaction

df.corr()[['meantempm']].sort_values('meantempm')  



predictors = ['meantempm_1',  'meantempm_2',  'meantempm_3', 'meantempm_4',  
              'mintempm_1',   'mintempm_2',   'mintempm_3', 'mintempm_4',
              'meandewptm_1', 'meandewptm_2', 'meandewptm_3', 'meandewptm_4',
              'maxdewptm_1',  'maxdewptm_2',  'maxdewptm_3', 'maxdewptm_4',
              'maxhumidity_1', 'maxhumidity_2', 'maxhumidity_3','maxhumidity_4',
              'minhumidity_1','minhumidity_2','minhumidity_3','minhumidity_4',
              'mindewptm_1',  'mindewptm_2',  'mindewptm_3', 'mindewptm_4',
              'maxtempm_1',   'maxtempm_2',   'maxtempm_3', 'maxtempm_3',
              'maxpressurem_1', 'maxpressurem_2' , 'maxpressurem_3', 'maxpressurem_4',
              'minpressurem_1', 'minpressurem_2', 'minpressurem_3', 'minpressurem_4',
              'precipm_1', 'precipm_2','precipm_3' ,'precipm_4',
              'sealevelpressure_1', 'sealevelpressure_2' ,'sealevelpressure_3' ,'sealevelpressure_4',
              'airtemperature_1', 'airtemperature_2', 'airtemperature_3','airtemperature_4',
              'dewpoint_1', 'dewpoint_2', 'dewpoint_3', 'dewpoint_4',
              'winddirection_1', 'winddirection_2', 'winddirection_3','winddirection_4',
              'windspeed_1', 'windspeed_2', 'windspeed_3','windspeed_4',
              ]

df2 = df[['meantempm'] + predictors] 


# separate our my predictor variables (X) from my outcome variable y
X = df2[predictors]  
y = df2['meantempm']

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X) 

X.iloc[:5, :5]  

#from ghere bakward leimination
# (1) select a significance value
alpha = 0.05

# (2) Fit the model
model = sm.OLS(y, X).fit()

# (3) evaluate the coefficients' p-values
model.summary()  

# (4) - Use pandas drop function to remove this column from X
q = True
while q:
    max_index, max_value = max(enumerate(model.pvalues), key=operator.itemgetter(1))
    if (max_value > 0.05):
        X = X.drop(model.pvalues.index[max_index], axis=1)
        model = sm.OLS(y, X).fit()
        print(model.summary())
    else:
        q = False

#code for backward
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


X_train = X.head(800)
X_test = X.tail(296)

y_train = y.head(800)
y_test = y.tail(296)

regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X_train, y_train)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

# Evaluate the prediction accuracy of the model

print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))  
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))  
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))

plt.plot(X_test.index[5:23], y_test[5:23], X_test.index[5:23], prediction[5:23])
