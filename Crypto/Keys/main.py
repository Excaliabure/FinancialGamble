import json
from mnemonic import Mnemonic
from hdwallet import HDWallet
from hdwallet.cryptocurrencies import Bitcoin as Cryptocurrency
from binascii import unhexlify

# Step 1: Generate 128 bits of cryptographically secure random entropy.
mnemo = Mnemonic("english")
entropy = mnemo.generate(128)


# entropy_bytes = unhexlify(entropy_hex)
# # Step 2: Generate a mnemonic (seed phrase) from the entropy.
mnemonic_phrase = entropy
print("1. Generated Seed Phrase:")
print(f"   '{mnemonic_phrase}'")
print("-" * 50)

# # Validate the generated mnemonic using the `mnemonic` library.
if mnemo.check(mnemonic_phrase):
    print("   ✅ Mnemonic is valid.")
else:
    print("   ❌ Mnemonic is not valid.")
    exit()

# # Step 3: Create the 512-bit master seed.
# # The `to_seed` function still uses PBKDF2.
passphrase = ""
master_seed = mnemo.to_seed(mnemonic_phrase, passphrase=passphrase)
print("\n2. Derived Master Seed (512 bits):")
print(f"   Hex: {master_seed.hex()}")
print("-" * 50)

# # Step 4: Create a Hierarchical Deterministic (HD) Wallet from the master seed.
hd_wallet = HDWallet(
    cryptocurrency="Bitcoin",
    seed=master_seed.hex()
)

print("\n3. Created HD Wallet Instance:")
print(f"   Cryptocurrency: {hd_wallet.cryptocurrency_name()}")
print("-" * 50)

# # Step 5: Derive a specific key pair and address using a BIP-44 path.
hd_wallet.from_path(path="m/44'/0'/0'/0/0")

private_key = hd_wallet.private_key()
public_key = hd_wallet.public_key()
address = hd_wallet.address()

