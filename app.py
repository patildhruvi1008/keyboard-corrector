import streamlit as st
import pandas as pd
import numpy as np
import textdistance
import re
from collections import Counter

# Title
st.title("Simple Autocorrect App")

# File uploader
uploaded_file = st.file_uploader("Upload a text file", type=['txt'])

if uploaded_file is not None:
    # Read file
    file_name_data = uploaded_file.read().decode('utf-8')
    file_name_data = file_name_data.lower()
    words = re.findall(r'\w+', file_name_data)

    # Vocabulary
    V = set(words)

    st.write(f"The first ten words in the text are: {words[:10]}")
    st.write(f"There are {len(V)} unique words in the vocabulary.")

    # Word frequency
    word_freq_dict = Counter(words)
    st.write("Top 10 most common words:")
    st.write(word_freq_dict.most_common(10))

    # Probabilities
    probs = {}
    Total = sum(word_freq_dict.values())
    for k in word_freq_dict.keys():
        probs[k] = word_freq_dict[k] / Total

    # Autocorrect function
    def my_autocorrect(input_word):
        input_word = input_word.lower()
        if input_word in V:
            return '✅ Your word seems to be correct.'
        else:
            similarities = [1 - textdistance.Jaccard(qval=2).distance(v, input_word) for v in word_freq_dict.keys()]
            df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
            df = df.rename(columns={'index': 'Word', 0: 'Prob'})
            df['Similarity'] = similarities
            output = df.sort_values(['Similarity', 'Prob'], ascending=False).head()
            return output

    # Text input for autocorrect
    input_word = st.text_input("Enter a word to autocorrect:")

    if input_word:
        result = my_autocorrect(input_word)
        if isinstance(result, str):
            st.success(result)
        else:
            st.write("Did you mean:")
            st.dataframe(result)

else:
    st.info("Please upload a `.txt` file to get started.")
