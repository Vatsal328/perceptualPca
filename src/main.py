import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Generate random weights (as float)
w = np.random.uniform(low=-1.0, high=1.0, size=5)
print("Weights :", w)

# Generate random integer values for x1, x2, x3, x4
num_samples = 5000 
x = np.random.randint(low=-1000, high=1000, size=(num_samples, 4))

# Calculate f(x) using the linear function
f_x = np.dot(x, w[1:]) + w[0]
labels = (f_x >= 0).astype(int)

# Create the data.txt
with open('B22CS059_data.txt', 'w') as file:
    file.write(str(num_samples) + '\n')
    for i in range(num_samples):
        row = ' '.join(map(str, x[i])) + ' ' + str(labels[i]) + '\n'
        file.write(row)
# Remove the empty line created at the end
with open('B22CS059_data.txt', 'rb+') as file:
    file.seek(-1, 2)
    if file.read(1) == b'\n':
        file.seek(-1, 2)
        file.truncate()
print("Data file created successfully.")

# Calculate percentage of positive and negative entries (for verification)
df = pd.read_csv('data.txt', delimiter=' ', header=None, skiprows=1)
pos_per = (df[4] == 1).mean()*100
neg_per = (df[4] == 0).mean()*100
print("Percentage of positive entries: {:.2f}%".format(pos_per))
print("Percentage of negative entries: {:.2f}%".format(neg_per))

def split(size):
    # Split the dataset
    x_train, x_test, labels_train, labels_test = train_test_split(x, labels, test_size=size, random_state=42)

    # Write train.txt
    with open('B22CS059_train.txt', 'w') as file:
        file.write(str(len(x_train)) + '\n')
        for i in range(len(x_train)):
            row = ' '.join(map(str, x_train[i])) + ' ' + str(labels_train[i]) + '\n'
            file.write(row)
    # Remove the empty line created at the end
    with open('B22CS059_train.txt', 'rb+') as file:
        file.seek(-1, 2)
        if file.read(1) == b'\n':
            file.seek(-1, 2)
            file.truncate()
    print("Train file created successfully.")

    # Write test.txt without labels
    with open('B22CS059_test.txt', 'w') as file:
        file.write(str(len(x_test)) + '\n')
        for i in range(len(x_test)):
            row = ' '.join(map(str, x_test[i])) + '\n'
            file.write(row)
    # Remove the empty line created at the end
    with open('B22CS059_test.txt', 'rb+') as file:
        file.seek(-1, 2)
        if file.read(1) == b'\n':
            file.seek(-1, 2)
            file.truncate()
    print("Test file created successfully.")

    # Write test_labels.txt (It will be used to check accuracy)
    with open('test_labels.txt', 'w') as file:
        for label in labels_test:
            file.write(str(label) + '\n')
    # Remove the empty line created at the end
    with open('test_labels.txt', 'rb+') as file:
        file.seek(-1, 2)
        if file.read(1) == b'\n':
            file.seek(-1, 2)
            file.truncate()
    print("Test labels file created successfully.")

split(0.3) # Set the size of the split