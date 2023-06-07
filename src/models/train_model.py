
import pandas as pd
from sklearn.model_selection import train_test_split

class TrainModel():

    def __init__(self, path):
        self.path = path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.path, sep=';')

    def get_data(self):
        return self.data

    def split_data(self):

        data = self.get_data()
        X = data.drop('diagnose', axis=1)
        y = data['diagnose']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        return X_train, X_test, y_train, y_test
