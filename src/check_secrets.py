# check_secrets.py
import yaml, os

path = os.path.join("config", "secrets.yaml")
data = yaml.safe_load(open(path))

print("Full content of secrets.yaml:\n", data, "\n")
print("Contains 'binance' key?", "binance" in data)
bc = data.get("binance", {})
print("  Contains api_key?", "api_key" in bc)
print("  Contains api_secret?", "api_secret" in bc)
