import pandas as pd
import numpy as np
import json

try:
	df = pd.read_csv("data.csv")
except FileNotFoundError:
	print("Error: 'data.csv' file not found.")
	exit(1)
except pd.errors.EmptyDataError:
	print("Error: 'data.csv' is empty.")
	exit(1)

try:
	df = df.dropna(subset=["km", "price"])
	if df.empty:
		print("Error: No valid data available after removing rows with missing values.")
		exit(1)
except KeyError:
	print("Error: One or more columns not found in 'data.csv'.")
	exit(1)

try:
	original_mileage = df["km"].astype(float).to_numpy()
	original_price = df["price"].astype(float).to_numpy()
except ValueError as e:
	print(f"Error: Non-numeric value found in CSV: {e}")
	exit(1)

mean_mileage = np.mean(original_mileage)
std_mileage = np.std(original_mileage)
mean_price = np.mean(original_price)
std_price = np.std(original_price)

# Standardize
mileage = (original_mileage - mean_mileage) / std_mileage
price = (original_price - mean_price) / std_price

theta0, theta1 = 0.0, 0.0
learning_rate = 0.01
max_iter = 100000
prev_loss = float('inf')
loss_history = []
theta0_list = []
theta1_list = []

np.seterr(over='raise', invalid='raise')
for i in range(max_iter):
	estimated = theta0 + theta1 * mileage
	errore = estimated - price

	grad_theta0 = np.mean(errore)
	grad_theta1 = np.mean(errore * mileage)

	theta0 -= learning_rate * grad_theta0
	theta1 -= learning_rate * grad_theta1

	loss = np.mean(errore**2)
	loss_history.append(loss)

	# De-standardize
	d_t1 = theta1 * (std_price / std_mileage)
	d_t0 = theta0 * std_price + mean_price - d_t1 * mean_mileage
	theta0_list.append(d_t0)
	theta1_list.append(d_t1)

	if abs(prev_loss - loss) < 1e-9 or (abs(grad_theta0) < 1e-6 and abs(grad_theta1) < 1e-6):
		print(f"Converged at iteration {i + 1}")
		break
	prev_loss = loss
else:
	print("Reached max iterations without full convergence")

# De-standardize
theta1 = theta1 * (std_price / std_mileage)
theta0 = theta0 * std_price + mean_price - theta1 * mean_mileage

with open("thetas.json", "w") as f:
	json.dump({"theta0": theta0, "theta1": theta1}, f)

history = {
    "theta0": theta0_list,
    "theta1": theta1_list,
    "loss": loss_history
}

with open("theta_history.json", "w") as f:
    json.dump(history, f)

print(f"Training complete. theta0 = {theta0}, theta1 = {theta1} saved to 'thetas.json'.")
