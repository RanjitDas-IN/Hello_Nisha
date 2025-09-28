import re
import pandas as pd
from g2p import make_g2p

# Load your CSV file
df = pd.read_csv("/home/ranjit/Desktop/projects/Hello_Nisha/1_unique_words.csv")  # <-- replace with your CSV path
text_column = "text" 

# Extract all words from the text column
all_words = []
for text in df[text_column].dropna():
    words = re.findall(r"\b\w+(?:'\w+)?\b", text)  # keeps "I'll" as one word
    all_words.extend(words)

# Get unique words
unique_words = sorted(set(all_words))

# Initialize the G2P transducer for ARPABET
transducer = make_g2p('dan', 'eng-arpabet')

# Prepare a dictionary to store word pronunciations
word_pronunciations = {}

# Process each unique word
for word in unique_words:
    # Get the transduction graph for the word
    graph = transducer(word)
    
    # Transduce to get the phoneme sequence
    arpabet = graph.transduce()
    
    # Check if the transduction is successful
    if arpabet:
        # Join the phonemes into a string and store in the dictionary
        word_pronunciations[word.upper()] = ' '.join(arpabet)
    else:
        # Handle cases where no pronunciation is found
        word_pronunciations[word.upper()] = 'N/A'

# Optionally, print out the word pronunciations
for word, pron in word_pronunciations.items():
    print(f"{word}: {pron}")
