from urllib.parse import quote_plus

# Replace "your_actual_password" with your real MongoDB password
password = "Jnn@1902"

# Encode the password
encoded_password = quote_plus(password)

print("=" * 50)
print("MongoDB Password Encoding Helper")
print("=" * 50)
print(f"Original password: {password}")
print(f"Encoded password:  {encoded_password}")
print("=" * 50)
print("\nCopy the ENCODED password and use it in your .env file:")
print(f"MONGO_CONNECTION_STRING=mongodb+srv://nulltrace:{encoded_password}@nulltrace.dlwpwnx.mongodb.net/threat_detector")
print("=" * 50)