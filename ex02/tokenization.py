"""Exercise 2: tokenize a text with spaCy.

Calling the model on a text returns a Doc, which is already a sequence of
tokens: no separate tokenizer call is needed, and the dots are tokens too.
"""
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Tokenize this sentence. And this one too."

doc = nlp(text)

if __name__ == "__main__":
    for token in doc:
        print(token)
