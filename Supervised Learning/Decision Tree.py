# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset iris (dataset bawaan scikit-learn)
iris = datasets.load_iris()
X = iris.data  # Menggunakan semua fitur
y = iris.target  # Label

# Membagi data menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat model Decision Tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)

# Melatih model dengan data training
clf.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = clf.predict(X_test)

# Menampilkan akurasi
print(f"Akurasi: {accuracy_score(y_test, y_pred)}")

# Visualisasi Decision Tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Decision Tree pada Dataset Iris')
plt.show()
