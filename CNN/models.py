import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx, embedding_strategy, pooling_chunk = 1, k_max = 1):
        super().__init__()

        self.embedding_strategy = embedding_strategy
        self.pooling_chunk = pooling_chunk
        self.k_max = k_max
        if embedding_strategy == 'multi':
            self.static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.non_static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.static_embedding.weight.requires_grad = False
        elif embedding_strategy == 'static':
            self.static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.static_embedding.weight.requires_grad = False
        else :
            self.non_static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        in_channels = 1
        if embedding_strategy == 'multi':
            in_channels = 2
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters * self.pooling_chunk * self.k_max, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        if self.embedding_strategy == 'multi':
            static_embedded = self.static_embedding(text)
            non_static_embedded = self.non_static_embedding(text)
            # embedded = [batch size, sent len, emb dim]
            embedded = torch.stack([static_embedded, non_static_embedded], dim=1)
        elif self.embedding_strategy == 'static':
            static_embedded = self.static_embedding(text)
            embedded = static_embedded.unsqueeze(1)
        else:
            non_static_embedded = self.non_static_embedding(text)
            embedded = non_static_embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [self.pooling(x=conv) for conv in conved]
        # pooled_n = [batch size, n_filters, pooling_chunk * k_max]
        pooled = [pool.view(pool.shape[0], pool.shape[1] * pool.shape[2]) for pool in pooled]
        # pooled_n = [batch size, n_filters * pooling_chunk * k_max]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes) * pooling_chunk]
        return self.fc(cat)

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def pooling(self, x):
        if self.k_max == 1:
            pooled = F.adaptive_max_pool1d(x, self.pooling_chunk)
            # pooled_n = [batch size, n_filters, pooling_chunk]
            return pooled
        if self.pooling_chunk == 1:
            pooled = self.kmax_pooling(x = x, dim = 2, k = self.k_max)
            # pooled_n = [batch size, n_filters, k_max]
            return pooled

        chunks = x.chunk(self.pooling_chunk, 2)
        pooled = torch.zeros(x.shape[0], x.shape[1], 0)
        pooled = pooled.cuda()
        for chunk in chunks:
            pooled = torch.cat((pooled, self.kmax_pooling(chunk, 2, self.k_max)), dim = 2)
        return pooled
