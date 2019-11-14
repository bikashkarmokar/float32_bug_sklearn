import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('sample_data_large_value.csv')

df = df.reset_index(drop=True)
max_value_df = df.max().max()
print("max_value_df before replace: {}".format(max_value_df))
list_cols_contain_max_value = df.columns[df.isin([max_value_df]).any()]
print(list_cols_contain_max_value)

float32_info = np.finfo(np.float32)
max_float_range = float32_info.max
print(max_float_range)

if max_value_df > max_float_range:
    print('DF contains value greater than max float32 value in columns' + str(list_cols_contain_max_value))

df_features = df.iloc[:, 0:-1]
df_target = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=.25, random_state=0)

model = AdaBoostRegressor()

try:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = r2_score(y_test, y_pred)
    print(acc)

except Exception as error:
    print(error)



