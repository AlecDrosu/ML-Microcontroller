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
    # add a title, and labels to the axes
    plt.title('Data from ' + csvFile)
    plt.xlabel(x)
    plt.ylabel('Data for ' + y1 + ', ' + y2 + ', and ' + y3)
    # Show the plot
    plt.show()

# Call the function (e.g. plotData('./Final data/preprocessed_files/1.csv', 'Time', 'Current', 'Voltage', 'Power'))