# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Contoh dataset (data buatan)
# Input (X) adalah data 1 dimensi, dan Output (y) adalah targetnya
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2.3, 2.9, 3.2, 4.4, 4.9, 6.1, 6.5, 7.8, 8.0, 9.2])

# Split data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linear
model = LinearRegression()

# Melatih model dengan data training
model.fit(X_train, y_train)

# Memprediksi data testing
y_pred = model.predict(X_test)

# Menampilkan koefisien dan intercept
print(f"Koefisien (Slope): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Plot data asli dan garis prediksi
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, model.predict(X), color='red', label='Garis Prediksi')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Menampilkan nilai prediksi untuk data testing
for i, (true, pred) in enumerate(zip(y_test, y_pred)):
    print(f"Data Test {i+1}: Asli = {true}, Prediksi = {pred}")
