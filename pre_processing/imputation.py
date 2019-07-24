import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

data = pd.read_csv("imputation_example.csv", header=None)
data_value = data.values
imputer = SimpleImputer(missing_values=np.nan, strategy="median")
imputer.fit(data_value)
data_value = imputer.transform(data_value)
print(data_value)

