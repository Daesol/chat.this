import pickle

def save_data(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load_data(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data
