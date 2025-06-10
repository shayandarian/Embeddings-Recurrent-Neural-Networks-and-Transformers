This project is from 2024.<br/>
The purpose of this project is to do sentiment analysis on comments left by students at Rate My Professors.<br/>
# Datasets:
## 1. Embeddings:
  - This project uses [GloVe](https://nlp.stanford.edu/projects/glove/), which is a pre-trained embedding.
  - To use GloVe from Keras, the dataset needs to be downloaded and then used to create an Embedding layer. See [Using pre-trained word embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/) in the Keras examples.
  - PyTorch, on the other hand, includes GloVe in the "torchtext.vocab" module. See [Pretrained Word Embeddings](https://docs.pytorch.org/text/stable/vocab.html#pretrained-word-embeddings).
  - Note: For this project, wget and unzip were not used to download datasets, as shown in the links above. Instead, a copy was uploaded to [Google Drive](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=c2W5A2px3doP), though it is also possible to [save and reload the model](https://www.tensorflow.org/guide/keras/serialization_and_saving) once the embedding layer has been created. Alternatively, a third-party library such as [Gensim](https://radimrehurek.com/gensim/) as another option that can be considered.
  - Since the exact GloVe dataset file used for this project was too big to upload directly to GitHub, a link to the file can be found [here](https://drive.google.com/file/d/1noXy3tqw2FI3QWe7W9Chodhk_eKOw721/view?usp=drive_link).
## 2. Comments and Ratings:
In addition to leaving comments on Rate My Professors, reviews also include scores for "Quality" and "Difficulty". There are two publicly available datasets that include this information:
  - A [larger dataset](https://data.mendeley.com/datasets/fvtfjyvw7d/2) in [CSV](https://docs.python.org/3/library/csv.html) format from [Dr. Hibo Je](https://data.mendeley.com/datasets/fvtfjyvw7d/2) at Tsinghua University.
  - A [smaller dataset](https://www.kaggle.com/datasets/tilorc/rate-my-professor-reviews-5c-colleges) in [JSON](https://docs.python.org/3/library/json.html) format from Kaggle containing reviews from the undergraduate Claremont Colleges.
In the future, if additional data should be used for training, both [data augmentation](https://neptune.ai/blog/data-augmentation-nlp) and [web scrapping code](https://pypi.org/project/RateMyProfessorAPI/) are approaches to be considered.

# Tasks:
## 1. Embeddings for the Dataset:
  - While words can be represented using [one-hot](https://en.wikipedia.org/wiki/One-hot) encoding, analysis can be found to be more effective with a [word embedding](https://en.wikipedia.org/wiki/Word_embedding). Both one-hot encoding and word embeddings were implemented in this project.
  - We began by using a pre-trained embeddign model, but once all the tasks for the project were completed, we experimented with fine-tuning the embedding layer for the listed tasks.
## 2. Recurrent Neural Network:
  - A recurrent neural network model was constructed to predict the quality and difficulty scores that a student will assign, given the text of the student's comments.
  - For a detailed description of the RNN architecture, the experiments done, the methods used, and the results found, see the RNN sections in the .ipynb file.
  - For Keras users, a good tutorial to get started with is [Text classification with an RNN](https://www.tensorflow.org/text/tutorials/text_classification_rnn).
  - For PyTorch users, a good turotial to get started with is [Text classification with the torchtext library](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html).
## 3. Transformer:
  - A network using Transformer encoder blocks instead of RNN layers was constructed for the same prediction task.
  - This network required position encoding for sequences in addition to word embedding.
  - For Keras users, the [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/) shows how to construct an encoder block from position embedding and multi-head attention layers, but [PositionEmbedding](https://keras.io/keras_hub/api/modeling_layers/position_embedding/) and [TransformerEncoder](https://keras.io/keras_hub/api/modeling_layers/transformer_encoder/) layers can be used directly.
  - For PyTorch users, the tutorial [Language Modeling with nn.Transformer and torchtext](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) shows how to construct both encoder and decoder blocks.
  - The performance of the Transformer was compared to the performance of the RNN.
