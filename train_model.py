from utils import create_training_data, store_processed_data, retrieve_processed_data
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

### Processing the training data
# Categories = ["circles", "squares", "triangles"]
# Directory = "Resources"
#
# training_data = create_training_data(Categories, Directory)
#
# X = []
# y = []
#
# for features, label in training_data:
#     features = np.array(features)
#     label = np.array(label)
#     X.append(features)
#     y.append(label)
#
# X = np.array(X, dtype=np.int8).reshape(-1, 56, 56, 1)
# y = np.array(y, dtype=np.int8)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# store_processed_data(X_train, X_test, y_train, y_test)

### The model - 99.71% accurate (1047/1050), inference time = 0.04 seconds
# if "X_train" not in locals():
#     X_train, X_test, y_train, y_test = retrieve_processed_data()
#
# model = keras.Sequential()
# model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(56, 56, 1), activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.Dense(3, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10)
#
# model.save('ShapeDetectionModel')

### Validating the model
if "model" not in locals():
    model = keras.models.load_model("ShapeDetectionModel")

if "X_train" not in locals():
    X_train, X_test, y_train, y_test = retrieve_processed_data()

class_names = ["Circle", "Square", "Triangle"]

correct = 0
wrong = 0

print("Testing model...")

for sample in range(0, len(X_test)):
    prediction = model.predict(X_test[sample].reshape(-1, 56, 56, 1))

    if class_names[np.argmax(prediction)] == class_names[y_test[sample]]:
        correct += 1
    else:
        wrong += 1

print("accuracy:" + str(100 * (correct / (correct + wrong))) + "%")
print("correct:" + str(correct))
print("wrong:" + str(wrong))
