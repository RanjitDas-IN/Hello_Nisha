import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/positive_mapping.psv", sep="|")

# Make sure the "graphs" folder exists
os.makedirs("graphs", exist_ok=True)

# Count samples per voice
voice_counts = df.drop(columns=["text"]).count().reset_index()
voice_counts.columns = ["voice", "count"]

# Plot
sns.barplot(
    data=voice_counts,
    x="voice",
    y="count",
    palette="tab10"   # You can try "tab20", "Set2", "pastel", etc.
)
plt.xticks(rotation=45, ha="right")
plt.title("Number of Audio Samples per Voice", color='red')
plt.xlabel("Voice Names", color='red')
plt.ylabel("Count of MP3 Files", color='red')


# Save inside "graphs" folder
plt.savefig("graphs/samples_per_voice.png", bbox_inches="tight")
plt.close()

# --------------------------------------------Text Length Distribution----------------------------------------------------------------------

# df["text_length"] = df["text"].str.len()

# sns.histplot(df["text_length"], bins=30, kde=True)
# plt.title("Distribution of Wake-up Text Lengths", color='red')
# plt.xlabel("Text Length (characters)", color='green')
# plt.ylabel("Frequency", color='green')

# plt.savefig("graphs/text_length_distribution.png", bbox_inches="tight")
# plt.close()
