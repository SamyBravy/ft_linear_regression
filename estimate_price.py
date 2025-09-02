import os
import json

if not os.path.exists("thetas.json"):
	print("thetas.json not found. Creating a default one")
	with open("thetas.json", "w") as f:
		json.dump({"theta0": 0, "theta1": 0}, f)

try:
    with open("thetas.json", "r") as f:
        data = json.load(f)
        theta0 = float(data["theta0"])
        theta1 = float(data["theta1"])
except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
    print("Error: Invalid thetas.json")
    exit(1)

print(f"Loaded theta0 = {theta0}, theta1 = {theta1}")

try:
	mileage = float(input("Enter mileage: "))
except ValueError:
	print("Invalid input. Enter a numeric value for mileage")
	exit(1)

estimated_price = theta0 + theta1 * mileage
print("Estimated price:", estimated_price)
