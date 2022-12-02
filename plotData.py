import pandas as pd
import matplotlib.pyplot as plt

# Create a function that plots four types of values on the same scatter plot for a given excel file
def plotData(excelFile):
    # Read data from the excel file
    data = pd.read_excel(excelFile)
    time = data['Time']
    current = data['Current']
    voltage = data['Voltage']
    power = data['Power']
    # plot time against voltage, current, and power on the same scatter plot with different colors
    plt.scatter(time, voltage, color='red', label='Voltage')
    plt.scatter(time, current, color='blue', label='Current')
    plt.scatter(time, power, color='green', label='Power')
    # Add a legend
    plt.legend()
    # Add labels to the axes
    plt.xlabel('Time')
    plt.ylabel('Voltage, Current, and Power')
    # Add a title
    plt.title('Voltage, Current, and Power vs Time')
    # Save the figure
    # plt.savefig('plotData.png')
    # Show the plot
    plt.show()