# NLP with spaCy

Solutions for the spaCy day of the AI piscine: the word embeddings, the
tokenization, the similarity between sentences, the named entities and the
part-of-speech tags.

## Structure

```
.
├── data/news_amazon.txt
├── ex00/
│   ├── requirements.txt
│   └── Notebook_ex00.ipynb
├── ex01/embedding.py
├── ex02/tokenization.py
├── ex03/embeddings_heatmap.py   -> cosine_distances.png
├── ex04/similarity.py
├── ex05/ner.py
└── ex06/pos_tags.py
```

## Virtual environment (ex00)

```bash
python -m venv env/ex00
source env/ex00/Scripts/activate      # Windows (Git Bash); Linux/macOS: source env/ex00/bin/activate
pip install -r ex00/requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python --version
python -c "import jupyter, pandas, spacy"
```

Tested with Python 3.13, spaCy 3.8.16 and the models `en_core_web_sm` and
`en_core_web_md` in version 3.8.0.

## Data

`news_amazon.txt` comes from the
[01-edu subject](https://github.com/01-edu/public/blob/master/subjects/ai/nlp-spacy/resources/news_amazon.txt).

## Running an exercise

```bash
python ex05/ner.py
```

## What an embedding is

An embedding turns a word into a vector, in a space where two close vectors
mean two close meanings. It replaces the Bag of Word of yesterday, whose
size grew with the vocabulary (10k columns and more), by a few hundred
values that carry the meaning.

The two models do not hold the same thing:

- **`en_core_web_sm`** has no real word vectors. The vector of a token comes
  from its `tok2vec` component and has **96** values. Asking it for a
  similarity prints the warning `W007`, which says exactly that; the subject
  asks for this model in ex01, ex02, ex04, ex05 and ex06.
- **`en_core_web_md`** carries real word vectors of **300** values, which is
  why ex03 uses it.

The embedding of a sentence is the average of the embeddings of its words,
and `similarity` is the cosine between two of those averages.

## Results

- **ex01** — the embedding of `car` has the shape **(96,)**.
- **ex02** — the 9 tokens match the audit exactly.
- **ex03** — every embedding has the shape **(300,)** and the heatmap shows
  the groups: laptop/computer, coffee/tea, dog/cat/kitty.
- **ex04** — the two sentences about shoes are the closest pair (0.73),
  then sentence 1 with sentence 3 (0.36), then sentence 2 with sentence 3
  (0.33). The order is the one of the audit.
- **ex05** — the named entities and the disambiguation match: the fruit
  `apple` is not an entity, the `Apple` of `Apple device` is an `ORG`.
- **ex06** — the four sentences mentioning Bezos as `NNP` match the audit
  exactly.

## Notes on the versions

The subject asks for spaCy 3.4.0 and the models 3.4.1. Neither has a wheel
for Python 3.13, so this uses spaCy 3.8.16 with the models 3.8.0. The
structure of the answers is the same, but the numbers that depend on the
weights of the model differ:

- **ex01** — the sum of the first 20 values of the embedding of `car` is
  `2.1977` here (`(9217763, 4194304)` as a ratio) against `(99959115,
  33554432)` in the audit. The shape, `(96,)`, is the same.
- **ex03** — the sum of the first 20 values of the embedding of `laptop` is
  `1.559` against `5.710` in the audit.
- **ex04** — the similarities are `0.73 / 0.36 / 0.33` against
  `0.71 / 0.43 / 0.33`. The ranking is unchanged.
- **ex05** — the entities are almost identical. This version adds
  `one CARDINAL` and misses `Facebook ORG` and `Apple II ORG`. The
  positions of ex05 question 2 are `Paul 0 4` and `Apple 49 54`, one
  character before the audit, whose text started with a space.
- **ex03** — in `en_core_web_md` 3.8.0, `dog` and `cat` have **the same
  vector**: their two rows of the table hold the same 300 values, so their
  cosine distance is 0.00 in the heatmap. The table of the model is pruned
  to 20000 vectors shared between many more words, which is what causes it.
  This is a property of the model, not of the code.
