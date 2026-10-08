"""Exercise 5: Named Entity Recognition.

The NER finds the names in a text and says what they are: a company (ORG), a
person (PERSON), a place (GPE)... It also disambiguates: in the second text
the fruit `apple` is not an entity at all, while the `Apple` of `Apple
device` is recognized as a company.
"""
import spacy

nlp = spacy.load("en_core_web_sm")

text = """Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops, and sells consumer electronics, computer software, and online services. It is considered one of the Big Five companies in the U.S. information technology industry, along with Amazon, Google, Microsoft, and Facebook.
Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 to develop and sell Wozniak's Apple I personal computer, though Wayne sold his share back within 12 days. It was incorporated as Apple Computer, Inc., in January 1977, and sales of its computers, including the Apple I and Apple II, grew quickly."""

ambiguous_text = "Paul eats an apple while watching a movie on his Apple device."

doc = nlp(text)
ambiguous_doc = nlp(ambiguous_text)

if __name__ == "__main__":
    print("1. the named entities and their label:")
    for entity in doc.ents:
        print(entity.text, entity.label_)

    print("\n2. disambiguation:")
    for entity in ambiguous_doc.ents:
        print(entity.text, entity.start_char, entity.end_char, entity.label_)
    print("\nThe first 'apple' (the fruit) is not an entity, the second one "
          "is recognized as a company.")
