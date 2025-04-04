import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X = np.array([
    [-3, 2], 
    [-6, 5],
    [3, -4],
    [2, -8]
])
print(X)
Y = np.array([1, 1, 2, 2])

plt.scatter(x = X[:, 0], y = X[:, 1], c = Y, cmap = 'Paired')
plt.show()

clf = SVC(gamma='auto')
clf.fit(X, Y)

Y_pred = clf.predict(X)
print(Y_pred == Y)

print(clf.predict([[5.4, 8.7]]))

x_min, x_max, y_min, y_max = (-6, 3, -8, 5)

xx = np.linspace(x_min, x_max, x_max-x_min+1)
yy = np.linspace(y_min, y_max, y_max-y_min+1)

xc, yc = np.meshgrid(xx, yy)
xc = xc.ravel()
yc = yc.ravel()

new_X = list(zip(xc, yc))
new_predicted_Y = clf.predict(new_X)

plt.scatter(xc, yc, c = new_predicted_Y)
plt.show()