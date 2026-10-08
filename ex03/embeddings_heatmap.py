"""Exercise 3: the embeddings of a sentence, and their distances.

`en_core_web_md` carries real word vectors of 300 dimensions. The heatmap
shows the cosine distance between every pair of words: the small squares of
the diagonal blocks (laptop/computer, coffee/tea/water/liquid, dog/cat/kitty)
are the words that belong to the same topic.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_distances

OUTPUT = Path(__file__).resolve().parent / "cosine_distances.png"

nlp = spacy.load("en_core_web_md")

text = "laptop computer coffee tea water liquid dog cat kitty"

doc = nlp(text)

# 1. one embedding of 300 values per word
words = [token.text for token in doc]
embeddings = np.array([token.vector for token in doc])

# 2. the distance between every pair of words
distances = cosine_distances(embeddings)

if __name__ == "__main__":
    print("1. embeddings")
    for token in doc:
        print(f"   {token.text:9s} shape: {token.vector.shape}")
    print("\n   sum of the first 20 values of the embedding of laptop:",
          doc[0].vector[:20].sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(distances, cmap="viridis")

    ax.set_xticks(range(len(words)), words, rotation=45, ha="right")
    ax.set_yticks(range(len(words)), words)
    ax.set_title("Pairwise cosine distances")
    fig.colorbar(image, ax=ax)

    # the value inside each square
    for i in range(len(words)):
        for j in range(len(words)):
            ax.text(j, i, f"{distances[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUTPUT)
    plt.show()
