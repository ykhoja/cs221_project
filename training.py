import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm

class Translator_data(object):
    
    def __init__(self, filename, periods):
        self.table = pd.read_csv(filename)
        self.periods = periods
        self.distributions = {}

    # Transforms the raw data table into a table with
    # the following columns: 'translator_id'(int), 'hour'(int), 'weekday'(bool), 'response'(str or NaN) 
    def DataPreprocessing(self):
        self.table = self.table[['translator_id','request_time','response']].copy()
        SATURDAY = 5
        timestamp = pd.to_datetime(self.table['request_time'])
        hours = timestamp.dt.hour
        days = timestamp.dt.weekday # returns number of day (Sat = 5, Sun = 6)
        self.table['hour'] = hours
        self.table['weekday'] = days < SATURDAY
        self.table = self.table[['translator_id','hour','weekday','response']]
    
    # Creates the rate of NONRESPONSE for each translator in the table
    # self.distributions is a dict with the translator_id as a key and
    # the tuple (rate per period(array of length: len(self.periods)), overall rate(float))
    def create_distribution(self):
        translators = self.table['translator_id'].unique()
        for translator in translators:
            rates = []
            overall = (1 + len(self.table[(self.table['translator_id'] == translator) & (self.table['response'] != 'yes')]))\
                                            *1./(2 + len(self.table[self.table['translator_id'] == translator]))
            for period in self.periods:
                minimum, maximum = period
                rates.append((len(self.table[(self.table['translator_id'] == translator) & (self.table['hour'] < maximum) & \
                                        (self.table['hour'] >= minimum) & (self.table['response'] != 'yes')])+1)* \
                                         1./(2 + len(self.table[(self.table['translator_id']== translator) & \
                                        (self.table['hour'] < maximum) & (self.table['hour'] >= minimum)])))
            self.distributions[translator] = (np.array(rates), overall)
        
    # Extracts the features for each row in the data
    # returns one row in the feature matrix X, and a class y
    def FeatureExtraction(self,row):
        translator = row['translator_id']
        hour = row['hour']
        weekday = float(row['weekday'])
        per_vec = np.array([float(minimum <= hour < maximum) for (minimum, maximum) in periods])
        overall_rate = self.distributions[translator][1]
        period_rate = np.dot(per_vec, self.distributions[translator][0])
        features = np.array([1.0,overall_rate,period_rate,weekday])
        y = float(row['response'] == 'yes')
        return features, y

def evaluate(prediction, label):
    return sum(prediction == label)* 1./len(label)
   
# periods = [(0,3),(3,6),(6,9),(9,12),(12,15),(15,18),(18,21),(21,24)]
periods = [(i,i+1) for i in range(24)]
# periods = [(0,6),(6,12),(12,18),(18,24)]
PT = Translator_data("PingedTranslators.csv", periods)
PT.DataPreprocessing()
PT.create_distribution()

X = np.zeros((len(PT.table), 4))
y = np.zeros(len(PT.table))
for index, row in PT.table.iterrows():
    X[index] = PT.FeatureExtraction(row)[0]
    y[index] = PT.FeatureExtraction(row)[1]

# Train / test splits
X_train, X_test, y_train, y_test = train_test_split(X,y)

X_test2 = X_test[y_test == 1]
y_test2 = y_test[y_test == 1]
X_test3 = X_test[y_test == 0]
y_test3 = y_test[y_test == 0]
X_test4 = np.concatenate((X_test2, X_test3[:len(X_test2)]), axis = 0)
y_test4 = np.concatenate((y_test2, y_test3[:len(y_test2)]), axis = 0)

# Train
classifier = lm.LogisticRegression(fit_intercept=False)
classifier.fit(X_train, y_train)

# Test
classifier.score(X_test4, y_test4)

# Class probabilities
classifier.predict_proba(X_test4)

