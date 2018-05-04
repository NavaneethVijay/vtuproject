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

df = pd.read_csv(r"Land_WD_WS.csv")


#tmp = df[['meantempm', 'meandewptm']].head(10)  

N = 1
features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",  
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm" ,"winddirection","windspeed"]

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

for sp in spread.index:
    plt.rcParams['figure.figsize'] = [14, 8]  
    print(sp)
    df[sp].hist() 
    plt.title('Distribution of maxhumidity_1')  
    plt.xlabel(sp)  
    plt.show()

for precip_col in ['precipm_1', 'precipm_2', 'precipm_3']:  
    # create a boolean array of values representing nans
    missing_vals = pd.isnull(df[precip_col])
    df[precip_col][missing_vals] = 0
    #re,moving all the NaN values
df = df.dropna()

#finding corelleaction

stats = df.corr()[['meantempm']].sort_values('meantempm') 

elements = []

for j,ele in enumerate(stats.meantempm):
    if ((abs(ele) > 0.5) and (stats.index[j] != 'meantempm')):
        elements.append(stats.index[j])
        print(stats.index[j])
        
        
        
predictors = elements

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
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


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

#plt.plot(X_test.index[60:90], y_test[60:90], 'r' ,  X_test.index[60:90], prediction[60:90])
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(y_test, label='actual values')
ax1.plot(prediction,label='predicted values')
plt.xlabel('Training dataset')  
plt.ylabel('Mean temperature') 
#plt.xlim(0.1,20)
plt.legend(loc='upper left');
plt.show()