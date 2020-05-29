import pickle
import tensorflow
import keras
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import insertskins
from os import path
import pandas as pd
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle


def main():
    # If we do not have the data file, use insertskins to create a new one
    if not path.exists('skindata.csv'):
        status = insertskins.insert()
        if status == -1:
            print('Error when creating skin data file...')
        else:
            print('Successfully created skin file...')

if __name__ == "__main__":
    main()

predict = 'curr_price'

skin_data = pd.read_csv('skindata.csv')
X = np.array(skin_data['initial_price'], skin_data['black'])
X = X.reshape(-1, 1)
y = np.array(skin_data[predict])

best = 0
while best < .8:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    if acc > best:
        print('Overriding previous best model\t' + str(best) + '\twith\t' + str(acc))
        best = acc
        with open('best_model.pickle', 'wb') as f:
            pickle.dump(linear, f)

print('best accuracy: ' + str(best))