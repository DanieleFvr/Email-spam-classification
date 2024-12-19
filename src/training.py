import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------- LOADING DATA

X_train = np.load("../data/X_train.npy")    # loading training features
X_test = np.load("../data/X_test.npy")      # loading testing features
y_train = np.load("../data/y_train.npy")    # loading training labels
y_test = np.load("../data/y_test.npy")      # loading testing labels

# --------------------------- BUILDING THE MODEL

# defining the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)), # first hidden layer
    tf.keras.layers.Dense(64, activation="relu"),   # second hidden layer
    tf.keras.layers.Dense(1, activation="sigmoid")  # output layer
])

# compiling the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# --------------------------- TRAINING THE MODEL

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# --------------------------- COMPUTING EVALUATION

y_predict = (model.predict(X_test) > 0.5).astype("int32")  # converting probabilities to binary predictions
accuracy = accuracy_score(y_test, y_predict)
F1 = f1_score(y_test, y_predict)
confusion = confusion_matrix(y_test, y_predict)

# printing evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"F1 score: {F1}")

# --------------------------- PLOTTING THE CONFUSION MATRIX

# plotting the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Greens", xticklabels=["Not spam", "Spam"], yticklabels=["Not spam", "Spam"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix")
plt.show()
