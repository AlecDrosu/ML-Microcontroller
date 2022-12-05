# Preprocess the data and save it to a new csv file
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

# Create a function that preprocesses the data and saves it to a new csv file, but this time, it also standardizes the data
def preprocessData(csvFile):
    # Read the csv file
    df = pd.read_csv(csvFile)
    # Extract the features, which are the current, voltage, and power
    x = df[['Current', 'Voltage', 'Power']]
    # Extract the target, which is the time
    y = df['Time']
    # Standardize the features
    x = std.fit_transform(x)
    # Create a new dataframe with the features and target
    df_new = pd.DataFrame(x)
    df_new['Time'] = y
    # Save the new dataframe to a new csv file
    df_new.to_csv('./Final data/preprocessed_files/' + csvFile.split('/')[-1], index = False)

# Call the function
for file in os.listdir('./Final data/csv_files'):
    if file.endswith('.csv'):
        preprocessData('./Final data/csv_files/' + file)