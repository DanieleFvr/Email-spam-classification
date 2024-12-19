# 1st NLP Assignment Report

> Favero Daniele (VR506229)
> Università di Verona, dipartimento di Informatica
> Corso di laurea magistrale in Artificial Intelligence
> A.A. 2024/25

## Objective
The goal of the project is to develop a binary classification system capable of discerning whether a given email falls in the category of *spam* or *not spam*.

## Chosen tools and technologies
### Programming language
Python
### Model architecture
The model consists of a multi-layer perceptron (**two-layer MLP**) with two fully connected hidden layers. The first and the second hidden layer are composed of 128 and 64 neurons respectively, and they both use the ReLU activation function. The output layer is composed of a single neuron and makes use of the sigmoid activation functions, which is perfect for a binary classification task such as this.
### Libraries
- *Pandas*, for reading and handling the tabular data contained in `emails_dataset.csv`
- *scikit-learn*, for preprocessing. In particular:
	- The `train_test_split()` method, for splitting the dataset into training and test sets
	- The `StandardScaler` function, for standardizing features
	- The `confusion_matrix`, `f1_score` and `accuracy_score` functions used during testing
- *Numpy*, for saving the training and testing data splits during preprocessing and accessing them during training
- *Tensorflow*, for building and training the MLP model
- *Matplotlib* and *Seaborn*, for visualizing the confusion matrix

## Performance
The model has the capabilities of consistently hitting an F1 and accuracy scores of 0.94 on the selected test data.
