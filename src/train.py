import sys
import numpy as np

class Perceptron:
    def __init__(self, num_features):
        self.weights = np.zeros(num_features + 1)  # Additional weight for bias term

    def predict(self, x):
        val = np.dot(self.weights[1:], x) + self.weights[0]
        return 1 if val >= 0 else 0

    # Function to train the model
    def train(self, X, y, lr=1, epochs=100):
        for _ in range(epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                self.weights[1:] += lr*(y[i] - prediction)*X[i]
                self.weights[0] += lr*(y[i] - prediction)

    # Function to save weights in a file
    def saveWeights(self, filename):
        np.savetxt(filename, self.weights, fmt='%f')
        with open('weights.txt', 'rb+') as file:
            file.seek(-1, 2)
            if file.read(1) == b'\n':
                file.seek(-1, 2)
                file.truncate()
        print("Training Over and Weights are saved")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use this command : python train.py train.txt")
        sys.exit(1)
    train_file = sys.argv[1]

    # Read data from train.txt
    data_train = np.loadtxt(train_file, skiprows=1)

    # Separate features and labels
    X_train = data_train[:, :-1]
    y_train = data_train[:, -1]

    # Normalize the data
    X_train_normalized = X_train / np.linalg.norm(X_train, axis=0)

    # Train the perceptron
    perceptron = Perceptron(num_features=X_train_normalized.shape[1])
    perceptron.train(X_train_normalized, y_train)

    # Save weights to file
    perceptron.saveWeights('weights.txt')