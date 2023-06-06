import os
import urllib.request

class Download():

    def __init__(self, destination_path):
        self.destination_path = destination_path

    def download(self):

        path = self.destination_path
        response = urllib.request.urlretrieve("https://www.opennn.net/files/breast_cancer.csv", os.path.join(path, "breast_cancer.csv"))

        return response

