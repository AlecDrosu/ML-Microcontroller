import os
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn import model_selection
import matplotlib.pyplot as plt


def save_regressor(regressor: GradientBoostingRegressor) -> None:
    with open('./trained_model/regressor.pickle', 'wb') as f:
        pickle.dump(regressor, f)


def load_regressor() -> GradientBoostingRegressor:
    with open('./trained_model/regressor.pickle', 'rb') as f:
        regressor = pickle.load(f)
    return regressor

paths = []

for file in os.listdir('./Final data/preprocessed_files'):
    if file.endswith('.csv'):
        paths.append('./Final data/preprocessed_files/' + file)

final_avg_accuracy = []
final_avg_std = []
accuracies_arr = []
std_arr = []

for path in paths[0:]:
    arr = np.array(pd.read_csv(path).to_numpy())

    accuracies = []

    for _ in range(10):
        np.random.shuffle(arr)
        X = arr[:, :-1]
        y = arr[:, -1]
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
        rgr = GradientBoostingRegressor(random_state=0)
        rgr.fit(X_train, y_train)

        y_pred = rgr.predict(X_test)
        accuracies.append(r2_score(y_test, y_pred))

    # make a plot of the current dataset, and how accurate the prediction is
    plt.scatter(arr[:, -1], rgr.predict(arr[:, :-1]), label='Prediction', s=2)
    plt.scatter(arr[:, -1], arr[:, -1], label='Actual', s=2)
    plt.legend()
    plt.grid(True, linestyle='--')
    #plt.title('Data from ' + path.split('/')[-1]), also remove .csv from the name
    plt.title('Data from ' + path.split('/')[-1][:-4])
    plt.xlabel('Actual Time')
    plt.ylabel('Predicted Time')
    plt.savefig('./plots/' + path.split('/')[-1].split('.')[0] + '.png')
    plt.clf()

    accuracies_arr.append(accuracies)
    std_arr.append(np.std(accuracies))
    
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    final_avg_accuracy.append(avg_accuracy)
    final_avg_std.append(std_accuracy)

    print("Average of accuracy on {} is {}%.".format(path.split("/")[-1], round(avg_accuracy * 100, 4)))
    print("Standard deviation of accuracy on {} is {}%.".format(path.split("/")[-1], round(std_accuracy * 100, 4)))

print("Final Accuracy of the model is {}%.".format(round(np.mean(final_avg_accuracy) * 100, 4)))
print("Final Standard Deviation of the model is {}%.".format(round(np.mean(final_avg_std) * 100, 4)))

# Create a boxplot to show the distribution of the accuracies
plt.boxplot(accuracies_arr)
plt.title('Distribution of the accuracies')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig('./plots/boxplot.png')
plt.clf()
# From the paths array, create a new array that contains the names of the files, and not the paths
files = []
for path in paths:
    files.append(path.split('/')[-1])

# create a table of final_avg_accuracy final_avg_std, and file names
df = pd.DataFrame({'File': files, 'Accuracy': final_avg_accuracy, 'Standard Deviation': final_avg_std})
# multiply the accuracy and standard deviation by 100 to get the percentage
df['Accuracy'] = df['Accuracy'] * 100
df['Standard Deviation'] = df['Standard Deviation'] * 100
# Add two more columns with the average accuracy and average std
df['Average Accuracy'] = np.mean(final_avg_accuracy) * 100
df['Average Standard Deviation'] = np.mean(final_avg_std) * 100
# save to xlsx file
df.to_excel('./Plots/Results.xlsx', index=False)
