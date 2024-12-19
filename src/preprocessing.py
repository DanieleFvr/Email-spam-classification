import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# --------------------------- LOADING DATA

data = pd.read_csv("../data/emails_dataset.csv")

# --------------------------- HANDLING MISSING VALUES

# checking for missing values
missingValues = data.isnull().sum().sum()

# deleting rows with any missing values
data = data.dropna()

# reporting deletion of rows with missing values
missingValues = data.isnull().sum().sum()
if missingValues > 0:
    print(f"{missingValues} have been found and their rows have been removed from the dataset.")
else:
    print("No missing values have been found in the dataset.")

# --------------------------- FEATURE EXTRACTION AND STANDARDIZATION

# separating features (X) and labels (y)
X = data.iloc[:, 1:-1]  # word count features
y = data.iloc[:, -1]    # target labels

# standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------- SPLITTING THE DATASET INTO TRAINING (80%) AND TESTING (20%) SETS

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7, stratify=y)

# saving the splits in the .npy format for model training
np.save("../data/X_train.npy", X_train)
np.save("../data/X_test.npy", X_test)
np.save("../data/y_train.npy", y_train)
np.save("../data/y_test.npy", y_test)

print("Data splits saved.")
