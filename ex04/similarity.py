"""Exercise 4: the similarity between two sentences.

The embedding of a sentence is the average of the embeddings of its words,
and `similarity` is the cosine between two of those averages. The two
sentences that ask for shoes are close even though they share few words,
which is what makes it possible to detect an intention.
"""
import spacy

nlp = spacy.load("en_core_web_sm")

sentence_1 = "I want to buy shoes"
sentence_2 = "I would love to purchase running shoes"
sentence_3 = "I am in my room"

doc_1, doc_2, doc_3 = nlp(sentence_1), nlp(sentence_2), nlp(sentence_3)

if __name__ == "__main__":
    print("sentence_1 <=> sentence 2 :", doc_1.similarity(doc_2))
    print("sentence_1 <=> sentence 3:", doc_1.similarity(doc_3))
    print("sentence_2 <=> sentence 3:", doc_2.similarity(doc_3))
