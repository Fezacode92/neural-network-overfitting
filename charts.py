from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, y = make_circles(n_samples=100, noise=0.35, factor=0.5, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

models = {
    "Underfitting": MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=5000, random_state=1),
    "Overfitting": MLPClassifier(hidden_layer_sizes=(200, 200, 200), max_iter=5000, random_state=1),
    "Good fit": MLPClassifier(hidden_layer_sizes=(16,), alpha=0.01, max_iter=5000, random_state=1),
}

xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k")
    train = model.score(X_train, y_train)
    test = model.score(X_test, y_test)
    ax.set_title(f"{name}\ntrain {train:.2f} / test {test:.2f}")

plt.savefig("fits.png")
plt.show()