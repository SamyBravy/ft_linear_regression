import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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
	mileage = df["km"].astype(float).to_numpy()
	price = df["price"].astype(float).to_numpy()
except ValueError as e:
	print(f"Error: Non-numeric value found in CSV: {e}")
	exit(1)

try:
	with open("thetas.json", "r") as f:
		data = json.load(f)
		theta0 = float(data["theta0"])
		theta1 = float(data["theta1"])
except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
	print("Error: Invalid or missing thetas.json")
	exit(1)

print(f"Loaded theta0 = {theta0}, theta1 = {theta1}")

estimated = theta0 + theta1 * mileage

mse = np.mean((estimated - price) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(estimated - price))
denom = np.sum((price - np.mean(price)) ** 2)
if denom == 0:
	r_squared = float('nan')
else:
	r_squared = 1 - (np.sum((price - estimated) ** 2) / denom)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r_squared}")

plt.scatter(mileage, price, color="blue", label="Data")
x_line = np.linspace(0, max(mileage) * 1.1, 100)
y_line = theta0 + theta1 * x_line
plt.plot(x_line, y_line, color="red", label="Regression")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.legend()
plt.title("Linear Regression Fit")
plt.savefig("regression_plot.png")
plt.close()

try:
	with open("theta_history.json", "r") as f:
		history = json.load(f)
		theta0_list = history["theta0"]
		theta1_list = history["theta1"]
		loss_history = history["loss"]
except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
	print("Error: Invalid or missing theta_history.json")
	exit(1)

if not (len(theta0_list) == len(theta1_list) == len(loss_history)):
	print("Error: theta history arrays have inconsistent lengths.")
	exit(1)

if len(theta0_list) == 0:
    print("Error: theta_history.json contains empty lists.")
    exit(1)

try:
    theta0_list = [float(x) for x in theta0_list]
    theta1_list = [float(x) for x in theta1_list]
    loss_history = [float(x) for x in loss_history]
except ValueError:
    print("Error: Non-numeric values found in theta_history.json.")
    exit(1)

plt.figure()
plt.plot(range(len(loss_history)), loss_history, color="green")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Evolution of Loss During Training")
plt.savefig("loss_plot.png")
plt.close()

fig, (ax_data, ax_loss) = plt.subplots(2, 1, figsize=(8, 10))
plt.subplots_adjust(bottom=0.25, hspace=0.3)

scat = ax_data.scatter(mileage, price, color="blue", label="Data")
line, = ax_data.plot([], [], color="red", label="Regression")
ax_data.set_xlabel("Mileage")
ax_data.set_ylabel("Price")
ax_data.set_title("Linear Regression Evolution")
ax_data.legend()

loss_line, = ax_loss.plot(loss_history, color="green", label="Loss")
if not loss_history:
	print("Error: Empty loss history.")
	exit(1)
loss_point, = ax_loss.plot(0, loss_history[0], 'ro')
ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("Mean Squared Error")
ax_loss.set_title("Loss evolution")
ax_loss.legend()

ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, 'Iteration', 0, len(theta0_list)-1, valinit=0, valstep=1)

def update(val):
	i = int(slider.val)
	x_line = np.linspace(0, max(mileage)*1.1, 100)
	y_line = theta0_list[i] + theta1_list[i]*x_line
	line.set_data(x_line, y_line)
	loss_point.set_data([i], [loss_history[i]])

	fig.canvas.draw_idle()

slider.on_changed(update)
update(0)
plt.show()
