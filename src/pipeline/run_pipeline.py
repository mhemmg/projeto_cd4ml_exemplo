import os

from sklearn.ensemble import RandomForestClassifier

from src.data.download_dataset import Download
from src.models.train_model import TrainModel

# Download data

absolute_path = os.path.abspath('')
data_raw_path = os.path.join(absolute_path, 'data/raw')

download = Download(destination_path = data_raw_path)
response = download.download()
data_path = response[0]

# load data
train_model = TrainModel(path = data_path)
train_model.load_data()

# processing data

# splitting data
X_train, X_test, y_train, y_test = train_model.split_data()

# training model
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

