import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from ann_visualizer.visualize import ann_viz


# Predict gender from NeuralHash

# fix random seed for reproducibility
np.random.seed(1)

# Read the csv file containing the dataset
csv_file = f"{os.getcwd()}/colorferet_metadataset.csv"

ordered_dataset = []
subject_indexes = {}

# Order data by subject
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    reader.__next__()
    i = 0
    for line in reader:
        subject = line[0].split('_')[0]
        nhash = [int(x) for x in line[1]]
        gender = 0 if line[2] == "Male" else 1
        ordered_dataset.append(nhash + [gender])
        if subject not in subject_indexes:
            subject_indexes[subject] = [i]
        else:
            subject_indexes[subject].append(i)

        i += 1

# Create new dataset list to be shuffled
dataset = [None] * len(ordered_dataset)
print(np.array(ordered_dataset))
print(np.array(ordered_dataset).shape)

subject_indexes_item_list = list(subject_indexes.items())
np.random.shuffle(subject_indexes_item_list)
subject_indexes = dict(subject_indexes_item_list)

# Set the ratio of training-to-testing samples. Splitratio denotes proportion of training samples
splitratio = 0.7

# Shuffle subjects in order to keep all images of the same subject in either the training or testing dataset
n = 0
for s, s_idxs in subject_indexes.items():
    for i in s_idxs:
        dataset[n] = ordered_dataset[i]
        n += 1

dataset = np.array(dataset)
print(dataset)
print(dataset.shape)

# Generate training and testing datasets
X_train = dataset[:int(len(dataset) * splitratio), 0:-1]
X_val = dataset[int(len(dataset) * splitratio):, 0:-1]
Y_train = dataset[:int(len(dataset) * splitratio), -1]
Y_val = dataset[int(len(dataset) * splitratio):, -1]
print(X_train)
print(X_train.shape)
print(Y_train)
print(Y_train.shape)

# Create model
model = Sequential()
model.add(Dense(256, input_dim=96, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f"./Graphs/InClass/{datetime.now().strftime('%Y%m%d-%H%M%S')}", write_graph=True)

# Fit the model
history = model.fit(X_train, Y_train, epochs=150, batch_size=10, validation_split=(1 - splitratio), callbacks=[tensorboard])

print("Model summary")
print(model.summary())

# Evaluate the model
scores = model.evaluate(X_val, Y_val)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Saving the model
model_json = model.to_json()
with open("model/dffn/gender_model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights("model/dffn/gender_model.h5")
print(f"Saved model to disk at {os.getcwd()}/model/dffn/gender_model.h5")

# Visualize the model
ann_viz(model, title="gender_model")
