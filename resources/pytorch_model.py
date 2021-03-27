import torch
import torch.nn as nn
from create_dataloader import AuthorDataset, AuthorBatchSampler
import csv
import numpy as np

class ContentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContentPredictor, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.rnn = nn.LSTM(input_dim, 256)
        self.predictor = nn.Sequential(
                nn.Linear(2 * 256, 256),
                nn.LayerNorm(256),
                nn.Tanh(),
                nn.Linear(256, output_dim),
                nn.Softmax(dim=-1),
        )

    def forward(self, comments):
        # comments: (batch, seq_len, input_size)
        comments = torch.transpose(comments, 0, 1)
        comments = self.norm(comments)
        # comments: (seq_len, batch, input_size)
        output, (h_0, c_0) = self.rnn(comments)
        # output: (seq_len, batch, input_size * num_directions)
        h_t = h_0[-1]
        c_t = c_0[-1]
        o_t = output[-1]
        x = torch.cat((h_t, c_t), dim=-1)
        return self.predictor(x)

N_EPOCHS = 100
print("[Loading Dataset]")
train_data = torch.load('train_dataloader.pt')
test_data = torch.load('test_dataloader.pt')

print("[Creating Model]")
model = ContentPredictor(128, 41)
# Remove this line if you don't have a CUDA-enabled GPU
model = model.to("cuda:0")
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
print("[Training Model]")
def do_epoch(model, dataset):
    total_loss = 0
    for i, batch in enumerate(dataset):
        optimizer.zero_grad()
        content, authors = batch
        # Remove these two lines if you don't have a CUDA-enabled GPU
        content = content.to("cuda:0")
        authors = authors.to("cuda:0")

        authors_p = model(content)
        loss = loss_fn(authors_p, authors)
        print(f"batch {i}/{len(dataset)}, loss={loss}", end='\r')
        total_loss += loss
        loss.backward()
        optimizer.step()

    return total_loss / len(dataset)

try:
    for i in range(N_EPOCHS):
        print(f"Epoch {i}")
        loss = do_epoch(model, train_data)
        print(f"final loss={loss}")
except KeyboardInterrupt:
    pass


preds = []
content_id = []
for batch in test_data:
    content, content_ids = batch
    # Remove these two lines if you don't have a CUDA-enabled GPU
    content = content.to("cuda:0")
    authors_p = model(content)
    authors_p = torch.argmax(authors_p, axis=-1)
    authors_p = authors_p.to("cpu").numpy()
    content_ids = content_ids.numpy()
    preds.extend(zip(content_ids, authors_p))

with open('preds.csv', 'w', newline='') as f:
    fout = csv.writer(f)
    fout.writerows(preds)
