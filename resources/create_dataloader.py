# For use with PyTorch
import csv
import nltk.data
from transformers import BertTokenizer
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# Parameters
# Modify to your liking
BATCH_SIZE = 64 # Size of each batch
EMBEDDING_SIZE = 128 # Size of generated Word2Vec Embeddings

class AuthorDataset(Dataset):
    def __init__(self, grouped_texts):
        super(Dataset, self).__init__()
        self.grouped_texts = {}
        self.grouped_authors = {}
        self._key_lengths = {}
        for k, v in grouped_texts.items():
            self.grouped_texts[k] = np.array(list(map(lambda pair: pair[0], v)))
            self.grouped_authors[k] = np.array(list(map(lambda pair: pair[1], v)))
            self._key_lengths[k] = len(v)

    @property
    def key_lengths(self):
        return self._key_lengths

    def __getitem__(self, index):
        length, within_group_index = index
        return (self.grouped_texts[length][within_group_index],
                self.grouped_authors[length][within_group_index])

    def __len__(self):
        return len(self.grouped_texts.keys())

class AuthorBatchSampler(Sampler):
    def __init__(self, key_lengths, batch_size, train=True):
        super(AuthorBatchSampler, self).__init__(None)
        self.train = train
        self.key_lengths = key_lengths
        self.batches = []

        for sentence_length, length in key_lengths.items():
            indices = [(sentence_length, i) for i in range(length)]
            np.random.shuffle(indices)
            self.batches.extend(indices[i: i + batch_size] for i in range(0, length, batch_size))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        batch_order = np.arange(len(self.batches))
        if self.train:
            batch_order = np.random.permutation(batch_order)

        for i in batch_order:
            yield self.batches[i]

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

    print("[Grouping by Sentence Length]")
    train_grouped_texts = defaultdict(list)
    for user_id, sentence in train_texts:
        encoded_sentence = np.array(tuple(model.wv[word] for word in sentence))
        train_grouped_texts[len(encoded_sentence)].append((encoded_sentence, user_id))

    test_grouped_texts = defaultdict(list)
    for content_id, sentence in test_texts:
        encoded_sentence = np.array(tuple(model.wv[word] for word in sentence))
        test_grouped_texts[len(encoded_sentence)].append((encoded_sentence, content_id))


    print("[Creating Training Dataset]")
    train_dataset = AuthorDataset(train_grouped_texts)
    train_sampler = AuthorBatchSampler(train_dataset.key_lengths, BATCH_SIZE)
    train_data = DataLoader(train_dataset, batch_sampler=train_sampler)
    print("[Creating Test Dataset]")
    test_dataset = AuthorDataset(test_grouped_texts)
    test_sampler = AuthorBatchSampler(test_dataset.key_lengths, BATCH_SIZE, train=False)
    test_data = DataLoader(test_dataset, batch_sampler=test_sampler)

    print("[Saving Training Dataset]")
    torch.save(train_data, 'train_dataloader.pt')
    print("[Saving Testing Dataset]")
    torch.save(test_data, 'test_dataloader.pt')


if __name__ == '__main__':
    main()

