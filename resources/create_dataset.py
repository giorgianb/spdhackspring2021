# For use with TensorFlow v2.0+
import csv
import nltk.data
from transformers import BertTokenizer
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np
import tensorflow as tf
import pickle

# Parameters
# Modify to your liking
EMBEDDING_SIZE = 128 # Size of generated Word2Vec Embeddings

def main():
    sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_texts = []
    print("[Tokenizing]")
    with open('train.csv') as f:
        fin = csv.reader(f)
        next(fin) # skip header
        for row in fin:
            user_id, text = row[0], row[1]
            for sentence in sentence_splitter.tokenize(text):
                train_texts.append((int(user_id), tokenizer.tokenize(sentence)))

    test_texts = []
    with open('test.csv') as f:
        fin = csv.reader(f)
        next(fin) # skip header
        for row in fin:
            content_id, text = row[0], row[1]
            for sentence in sentence_splitter.tokenize(text):
                test_texts.append((int(content_id), tokenizer.tokenize(sentence)))

    users = set(pair[0] for pair in train_texts)
    print(f"number users {len(users)}")

    sentences = tuple(pair[1] for pair in train_texts) + tuple(pair[1] for pair in test_texts)
    print("[Training Word2Vec Model]")
    model = Word2Vec(sentences=sentences, size=EMBEDDING_SIZE, window=8, min_count=1, iter=16)

    print("[Creating Dataset]")
    train_tensors = [([model.wv[w] for w in s], u) for u, s in train_texts]
    test_tensors = [([model.wv[w] for w in s], u) for u, s in test_texts]
    print('[Saving Training Dataset Tensors]')
    with open('train_dataset_tensors.pickle', 'wb') as fout:
        pickle.dump(train_tensors, fout)

    print('[Saving Test Dataset Tensors]')
    with open('test_dataset_tensors.pickle', 'wb') as fout:
        pickle.dump(test_tensors, fout)

if __name__ == '__main__':
    main()
