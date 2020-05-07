import argparse
import torch
from torchtext import data
from torchtext import datasets
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import models
import utils

# parse arguments
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--embedding_strategy', default='multi', type=str, help='determine embedding strategy' )
parser.add_argument('--pooling_chunk',default=1, type=int, help='number of split of feature map')
parser.add_argument('--k_max',default=1, type=int, help='select the k max feature while pooling')
arg = parser.parse_args()

# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setting how split validation
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# load data
TEXT = data.Field(tokenize='spacy', batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

# setting hyper parameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

# construct model
model = models.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX,
            embedding_strategy = arg.embedding_strategy,
            pooling_chunk = arg.pooling_chunk, k_max = arg.k_max)

# display parameter numbers
print(f'The model has {utils.count_parameters(model):,} trainable parameters')

# setting embedding
pretrained_embeddings = TEXT.vocab.vectors
if arg.embedding_strategy == 'multi':
    model.static_embedding.weight.data.copy_(pretrained_embeddings)
    model.non_static_embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.static_embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.static_embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    model.non_static_embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.non_static_embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
elif arg.embedding_strategy == 'static':
    model.static_embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.static_embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.static_embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
else:
    model.non_static_embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.non_static_embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.non_static_embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



# setting optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# setting loss function
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

# setting some vars
N_EPOCHS = 20
best_valid_loss = float('inf')
last_valid_loss = float('inf')

model = model.to(device)

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = utils.train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = utils.evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'CNN-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    if valid_loss >= last_valid_loss:
        break
    else:
        last_valid_loss = valid_loss

test_loss, test_acc, precision, recall, F1 = utils.test_evaluate(model, test_iterator, criterion)
print(f'Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc * 100:.2f}% |  Test. precision: {precision * 100:.2f}%'
      f'|  Test. recall: {recall * 100:.2f}% |  Test. F1: {F1 * 100:.2f}')

