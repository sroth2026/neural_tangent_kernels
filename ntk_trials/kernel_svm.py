import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.0,  # increase this to make it easier, decrease to make it harder
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_models = {}
for kernel in kernels:
    clf = SVC(kernel=kernel, degree=3, C=1.0)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    svm_models[kernel] = {
        'model': clf,
        'report': report
    }


def plot_decision_boundary(clf, X, y, ax, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_title(title)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()


def main():
    for i, kernel in enumerate(kernels):
        model = svm_models[kernel]['model']
        plot_decision_boundary(model, X_train_scaled, y_train, axes[i], f'SVM with {kernel} kernel')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()