import pickle
import gzip
import numpy as np

def load_data():
    # Correct the path to the dataset file if necessary
    path = '../data/mnist.pkl.gz'
    
    try:
        with gzip.open(path, 'rb') as f:
            # Ensure using 'latin1' encoding to handle potential compatibility issues
            training_data, validation_data, test_data = pickle.load(f, encoding='unicode-escape')
    except FileNotFoundError:
        print(f"File not found: {path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise
    
    return (training_data, validation_data, test_data)

def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)

# Example usage:
if __name__ == "__main__":
    try:
        training_data, validation_data, test_data = load_data_wrapper()
        print("Data loaded successfully!")
        # Add any further processing or testing here
    except Exception as e:
        print(f"Failed to load data: {e}")
