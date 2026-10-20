import pandas as pd

data = pd.read_csv("/kaggle/input/datasets/rohankayan/years-of-experience-and-salary-dataset/Salary_Data.csv")
print(data.head())
print(data.info())