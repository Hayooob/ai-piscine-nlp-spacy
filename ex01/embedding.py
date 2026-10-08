"""Exercise 1: load an embedding with spaCy.

An embedding turns a word into a vector of numbers, in a space where close
vectors mean close meanings. `en_core_web_sm` has no real word vectors: the
vector of a token comes from its `tok2vec` component and has 96 dimensions.
"""
import spacy

nlp = spacy.load("en_core_web_sm")

# 1. the embedding of a single word
car = nlp("car")
embedding = car.vector

# 2. the sum of its first 20 values, as an exact fraction
first_20_sum = float(embedding[:20].sum())

if __name__ == "__main__":
    print("1. shape:", embedding.shape)
    print("2. sum of the first 20 values:", first_20_sum)
    print("   as_integer_ratio():", first_20_sum.as_integer_ratio())
