import string
import random

# Create a pool of characters
char_pool = (
    string.ascii_letters  # A-Z, a-z
)

# Generate a list of unique characters
unique_chars = random.sample(char_pool, 52)
unique_chars.sort()
# print(unique_chars)