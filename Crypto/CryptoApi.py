import requests
import json
def init():

    return 

def fetch_new_crypto_coins():
    # CoinGecko API endpoint for fetching new crypto coins
    url = "https://api.coingecko.com/api/v3/coins/list"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Sort coins by addition date (newest first, if available)
        # Note: The CoinGecko API does not directly provide a creation or addition date,
        # so you might need to filter based on other criteria if required.

        print(f"Fetched {len(data)} cryptocurrencies.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []

def save_to_file(data, filename="new_crypto_coins.json"):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Saved data to {filename}")

def main():
    print("Fetching new cryptocurrency coins...")
    new_crypto_coins = fetch_new_crypto_coins()

    if new_crypto_coins:
        print(f"Successfully fetched {len(new_crypto_coins)} cryptocurrencies.")
        save_to_file(new_crypto_coins)
    else:
        print("No data fetched.")
