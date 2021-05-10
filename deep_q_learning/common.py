import pickle


def save_pickle(path, config):
    with open(path, 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
