import pandas as pd
import matplotlib.pyplot as plt
import os

# Create a function that plots four types of values on the same scatter plot for a given csv file
def plotData(csvFile, x, y1, y2, y3):
    # Read the csv file
    df = pd.read_csv(csvFile)
    # Create a scatter plot
    plt.scatter(df[x], df[y1], label = y1, s = 2)
    plt.scatter(df[x], df[y2], label = y2, s = 2)
    plt.scatter(df[x], df[y3], label = y3, s = 2)
    # Add a legend
    plt.legend()
    # add a grid to the plot with dashed lines
    plt.grid(True, linestyle = '--')
    # add a title that removes the .csv from the name and the path
    plt.title('Data from ' + csvFile.split('/')[-1][:-4])
    plt.xlabel(x)
    plt.ylabel('Data for ' + y1 + ', ' + y2 + ', and ' + y3)

for file in os.listdir('./Final data/csv_files'):
    if file.endswith('.csv'):
        plotData('./Final data/csv_files/' + file, 'Time', 'Current', 'Voltage', 'Power')
        plt.savefig('./Plots_Original_Data/' + file.split('/')[-1].split('.')[0] + '.png')
        plt.clf()