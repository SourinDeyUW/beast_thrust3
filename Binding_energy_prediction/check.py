from sklearn.metrics import mean_absolute_error
import pandas as pd
df=pd.read_csv("v1__test_300_epochs.csv")
sc=mean_absolute_error(df['y_test'],df['y_pred'])
print(sc)
