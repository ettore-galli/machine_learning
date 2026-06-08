# !pip install gensim nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

sample = "Word embeddings are dense vector representations of words."
tokenized_corpus = word_tokenize(sample.lower())

skipgram_model = Word2Vec(
    sentences=[tokenized_corpus],
    vector_size=100,
    window=5,
    sg=1,
    min_count=1,
    workers=4,
)

# Training
skipgram_model.train([tokenized_corpus], total_examples=1, epochs=10)
skipgram_model.save("skipgram_model.model")
loaded_model = Word2Vec.load("skipgram_model.model")
vector_representation = loaded_model.wv["word"]
print("Vector representation of 'word':", vector_representation)
