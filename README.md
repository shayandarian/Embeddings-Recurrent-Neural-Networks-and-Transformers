This project is from 2024.<br/>
The purpose of this project is to do sentiment analysis on comments left by students at Rate My Professors.<br/>
# Datasets:
## 1. Embeddings:
  - This project uses [GloVe](https://nlp.stanford.edu/projects/glove/), which is a pre-trained embedding.
  - To use GloVe from Keras, the dataset needs to be downloaded and then used to create an Embedding layer. See [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/) in the Keras examples.
  - PyTorch, on the other hand, includes GloVe in the "torchtext.vocab" module. See [Pretrained Word Embeddings](https://docs.pytorch.org/text/stable/vocab.html#pretrained-word-embeddings).
  - Note: For this project, wget and unzip were not used to download datasets, as shown in the links above. Instead, a copy is uploaded to [Google Drive](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=c2W5A2px3doP), though it is also possible to [save and reload the model](https://www.tensorflow.org/guide/keras/serialization_and_saving) once the embedding layer has been created. Alternatively, a third-party library such as [Gensim](https://radimrehurek.com/gensim/) as another option that can be considered.
## 2. Comments and Ratings:
