                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              # Importing the necessary libraries
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
# Load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
# Identify missing data (assumes that missing data is represented as NaN)
missing_data = dataset.isnull().sum()
# Print the number of missing entries in each column
print(missing_data)
# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(strategy='mean')
# Fit the imputer on the DataFrame
numerical_columns = dataset.select_dtypes(include=[np.number]).columns
imputer.fit(dataset[numerical_columns])
# Apply the transform to the DataFrame
dataset_imputed = pd.DataFrame(imputer.transform(dataset[numerical_columns]), columns=numerical_columns)
dataset[numerical_columns] = dataset_imputed
#Print your updated matrix of features
print(dataset)