import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from src.DatasetGenerator import U3Dataset

# Get New Saving Directory (dataset will be saved there)
reading_file = open("../outputs/config.txt", 'r')
number = reading_file.readline()
reading_file.close()
os.mkdir("../outputs/{}".format(number))
writing_file = open("../outputs/config.txt", 'w')
writing_file.write(str(int(number) + 1))
writing_file.close()
save_dir = "../outputs/{}".format(number)

# Generate Dataset
df = U3Dataset(save_dir=save_dir)
# df = RXDataset(save_dir=save_dir)

# Regression Task


X = df[['theta', 'phi', 'lam', 'E']]  # use this for U3
# X = df[['rx_theta', 'E']] # use this for RX
y = df[['p']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

kneighbor_regression = KNeighborsRegressor(n_neighbors=1)
kneighbor_regression.fit(X_train, y_train)
y_pred_train = kneighbor_regression.predict(X_train)

plt.plot(X_train, y_train, 'o', label="data", markersize=10)
plt.plot(X_train, y_pred_train, 's', label="prediction", markersize=4)
plt.legend();
plt.show()

y_pred_test = kneighbor_regression.predict(X_test)

plt.plot(X_test, y_test, 'o', label="data", markersize=8)
plt.plot(X_test, y_pred_test, 's', label="prediction", markersize=4)
plt.legend();
plt.show()

print(kneighbor_regression.score(X_test, y_test))
