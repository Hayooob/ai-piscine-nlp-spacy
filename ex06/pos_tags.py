"""Exercise 6: the Part-of-speech tags.

The POS tag says what a word is in its sentence: a verb, a noun, a proper
noun... `pos_` gives the coarse tag (PROPN) and `tag_` the detailed one
(NNP, a singular proper noun). Here it is used to find every sentence where
Bezos appears as a proper noun.
"""
from pathlib import Path

import spacy

DATA = Path(__file__).resolve().parent.parent / "data" / "news_amazon.txt"

nlp = spacy.load("en_core_web_sm")

text = DATA.read_text(encoding="utf-8")
doc = nlp(text)

if __name__ == "__main__":
    for sentence in doc.sents:
        for token in sentence:
            if token.text == "Bezos" and token.tag_ == "NNP":
                print("INFO: ", token.text, token.pos_, token.tag_)
                print("Sentence: ", sentence.text, end="\n\n\n")
                break
