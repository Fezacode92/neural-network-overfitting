from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=100, noise=0.35, factor=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = MLPClassifier(hidden_layer_sizes=(16,), alpha=0.01, max_iter=5000)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))