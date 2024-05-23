import sys
import numpy as np

# Function for prediction
def predict(weights, x):
    val = np.dot(weights[1:], x) + weights[0]
    return 1 if val >= 0 else 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use this command : python test.py test.txt")
        sys.exit(1)
    test_file = sys.argv[1]

    # Load weights from file
    weights = np.loadtxt('weights.txt')

    # Load test data from file
    data_test = np.loadtxt(test_file, skiprows=1)
    X_test = data_test
    X_test_normalized = X_test/np.linalg.norm(X_test, axis=0)

    # Predictions on test data
    predictions = [predict(weights, x) for x in X_test_normalized]

    # Save predictions to a txt file
    np.savetxt('predictions.txt', predictions, fmt='%d')
    with open('predictions.txt', 'rb+') as file:
        file.seek(-1, 2)
        if file.read(1) == b'\n':
            file.seek(-1, 2)
            file.truncate()

    # Convert predictions to a comma-separated string of 0s and 1s and print it
    predictions_str = ','.join(map(str, predictions))
    print(predictions_str)
