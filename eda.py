import random
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import Counter

# Load the dataset
ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")

# Randomly sample 5000 rows
sampled_data = random.sample(list(ds), 5000)

# Count how many passage_texts there are per row
passage_counts = [len(row["passages"]["passage_text"]) for row in sampled_data]

# Get frequency of each passage count
count_freq = Counter(passage_counts)

# Sort the counter for nicer plotting
sorted_counts = sorted(count_freq.items())
x, y = zip(*sorted_counts)

# Plot the bar graph
plt.figure(figsize=(12, 6))
plt.bar(x, y)
plt.xlabel("Number of Passages per Row")
plt.ylabel("Frequency")
plt.title("Distribution of Passage Counts in Sampled MS MARCO Rows")
plt.grid(True)
plt.show()
