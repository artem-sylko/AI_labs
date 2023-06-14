import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing()

X, y = shuffle(housing_data.data, housing_data.target)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print('\nADABOOST REGRESSOR')
print('Mean squared error = ', round(mse, 2))
print('Explained variance score = ', round(evs, 2))

feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names
feature_importances = 100.0*(feature_importances/max(feature_importances))

index_sorted = np.flipud(np.argsort(feature_importances))
pos = np.arange(index_sorted.shape[0]) + 0.5

print(feature_names)
print( feature_importances[index_sorted])

plt.figure(figsize=(10,6))
plt.bar(pos, feature_importances, align="center")
plt.xticks(pos, feature_names)
plt.ylabel("Relative importance")
plt.title("Оцінка важливості ознак з використанням AdaBoost")
plt.show()