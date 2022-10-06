""" Model IO with Pickle

pickle.dump(model, open(file_path, 'wb'))
loaded_model = pickle.load(open(file_path, 'rb'))

source: 
    https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
"""
import pickle

def save_scikit_model(model, file_path):
    pickle.dump(model, open(file_path, 'wb'))
    return

def load_scikit_model(file_path):
    loaded_model = pickle.load(open(file_path, 'rb'))
    return

