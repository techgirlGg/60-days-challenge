# Function to tokenize a paragraph
def tokenize_text(text):
    # Convert to lowercase and split into words
    tokens = text.lower().split()
    return tokens


# Function to count word frequency
def word_frequency(tokens):
    freq = {}
    for word in tokens:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq


# Function to remove stopwords
def remove_stopwords(tokens):
    stopwords = {
        "is", "am", "are", "was", "were", "the", "a", "an",
        "and", "or", "but", "if", "in", "on", "at", "to",
        "for", "of", "with", "by", "this", "that", "it"
    }
    
    filtered_tokens = []
    for word in tokens:
        if word not in stopwords:
            filtered_tokens.append(word)
    
    return filtered_tokens


# -------- Example Usage --------
paragraph = "This is a simple example to demonstrate text processing in Python"

tokens = tokenize_text(paragraph)
print("Tokens:", tokens)

filtered_tokens = remove_stopwords(tokens)
print("After removing stopwords:", filtered_tokens)

frequency = word_frequency(filtered_tokens)
print("Word Frequency:", frequency)