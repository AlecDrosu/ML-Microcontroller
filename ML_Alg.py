import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn import model_selection

paths = []

for file in os.listdir('./Final data/csv_files'):
    if file.endswith('.csv'):
        paths.append('./Final data/csv_files/' + file)

# read dataset into Pandas DataFrame with the names of the columns, where the first column is time, third is current, fourth is voltage, and fifth is power
df = pd.read_csv(paths[0])
for path in paths[1:]:
    pd.concat([df, pd.read_csv(path, names=['Time', 'Current', 'Voltage', 'Power'])])
# extracct the features, which are the current, voltage, and power
x = df[['Current', 'Voltage', 'Power']]
# extract the target, which is the time
y = df['Time']

x_arr = np.array(x)
y_arr = np.array(y)
x_arr = x_arr / x_arr.sum(axis=0).reshape(1, -1)
y_arr = y_arr / y_arr.sum(axis=0).reshape(1, -1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(x_arr, y.T, test_size=0.2, random_state=0)

rgr = GradientBoostingRegressor(random_state=0)
rgr.fit(X_train, y_train)

y_pred = rgr.predict(X_test)
accuracy = r2_score(y_test, y_pred)

print("The accuracy of the model is {}%.".format(round(accuracy, 2)))

