import pandas as pd
import re
from collections import Counter

# Load dataset
df = pd.read_csv("../data/raw/http_2025-03-15_to_2025-04-09.csv")

# Preprocessing function to tokenize text
def tokenize(text):
    """Splits text into lowercase words, removing special characters."""
    return re.findall(r"\b\w+\b", str(text).lower())

# Compute frequency distribution for `http_uri`
uri_words = Counter()
for uri in df["http_uri"].dropna():
    uri_words.update(tokenize(uri))

# Compute frequency distribution for `http_body`
body_words = Counter()
for body in df["http_body"].dropna():
    body_words.update(tokenize(body))

# Define a frequency threshold
FREQUENCY_THRESHOLD = 50  # Adjust this value based on your dataset

# Filter sensitive words based on the threshold
sensitive_words = {word: count for word, count in uri_words.items() if count >= FREQUENCY_THRESHOLD}
sensitive_words.update({word: count for word, count in body_words.items() if count >= FREQUENCY_THRESHOLD})

# Save the sensitive dictionary with counts to a file
with open("../data/trie/sensitive_words_with_counts.txt", "w") as f:
    for word, count in sorted(sensitive_words.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{word}: {count}\n")

print(f" Sensitive dictionary created with {len(sensitive_words)} words.")