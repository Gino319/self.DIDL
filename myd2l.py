import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
import sys
import collections
import re
import random
import math
import os

myd2l = sys.modules[__name__]



## Plot

def build_ax(xlabel=None, ylabel=None, xscale='linear', yscale='linear', xlim=None, ylim=None, figsize=(6, 3), grid=True):
    _, ax = plt.subplots(figsize=figsize)
    ax.grid(grid)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return ax

def show_list_len_pair_hist(legend, xlist, ylist, xlabel, ylabel):
    ax = myd2l.build_ax(xlabel, ylabel, grid=False)
    _, _, patches = ax.hist([[len(line) for line in xlist], [len(line) for line in ylist]])
    for patch in patches[1].patches:
        patch.set_hatch('/')
    ax.legend(legend)



## Classfication

def train_clf(net, n_epochs, lr, train_iter, test_iter, device=None):
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    if not device:
        device = next(net.parameters()).device
    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    losses, train_accs, test_accs = [], [], []
    time_sum = 0.0
    for epoch in range(n_epochs):
        net.train()
        loss_sum = 0.0
        start = time.time()
        for i, (X, y) in enumerate(train_iter):
            start = time.time()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            loss_sum += l.item()

        time_sum += time.time() - start
        losses.append(loss_sum / len(train_iter))
        train_accs.append(evaluate_accuracy(net, train_iter, device))
        test_accs.append(evaluate_accuracy(net, test_iter, device))

        print(f'epoch {epoch + 1: 2d}, loss {losses[-1]: .4f}, train acc {train_accs[-1]: .2f}, test acc {test_accs[-1]: .2f}')
    
    plot_mlp(n_epochs, losses, train_accs, test_accs)

    print(f'loss {losses[-1]: .4f}, train acc {train_accs[-1]: .2f}, test acc {test_accs[-1]: .2f}')
    print(f'{n_epochs * len(train_iter) * train_iter.batch_size / time_sum: .1f} examples/sec on {str(device)}')

def evaluate_accuracy(net, data_iter, device=None):
    if not device:
        device = next(net.parameters()).device
    net = net.to(device)
    
    net.eval()
    acc_sum = 0.0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            acc_sum += (y_hat.argmax(dim=1) == y).float().mean().item()
    
    return acc_sum / len(data_iter)

def plot_mlp(n_epochs, losses, train_accs, test_accs):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.grid(True)
    ax.set_xlabel('epoch')
    ax.plot(range(1, n_epochs + 1), losses, label='loss')
    ax.plot(range(1, n_epochs + 1), train_accs, label='train acc', c='g', linestyle='--')
    ax.plot(range(1, n_epochs + 1), test_accs, label='test acc', c='purple', linestyle='-.')

    plt.legend()
    plt.show()

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    minist_train = Datasets.FashionMNIST(root='E:/Datasets/FashionMNIST', download=False, train=True, transform=trans)
    minist_test = Datasets.FashionMNIST(root='E:/Datasets/FashionMNIST', download=False, train=False, transform=trans)
    train_iter = Data.DataLoader(minist_train, batch_size=batch_size, shuffle=True)
    test_iter = Data.DataLoader(minist_test, batch_size=batch_size, shuffle=True)

    return train_iter, test_iter



## Vocab

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= param.grad * lr / batch_size
            param.grad.zero_()

def read_time_machine(path='E:/Datasets/timemachine/timemachine.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='char'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Unknown token type')

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, min_freq=2, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        else:
            return [self.__getitem__(token) for token in tokens]
        
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        else:
            return [self.to_tokens(index) for index in indices]
    
    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def load_corpus_time_machine(path='E:/Datasets/timemachine/timemachine.txt', max_tokens=-1):
    lines = read_time_machine(path)
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[: max_tokens]
    
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, n_steps):
    corpus = corpus[random.randint(0, n_steps - 1):]
    n_subseqs = (len(corpus) - 1) // n_steps
    initial_indices = list(range(0, n_subseqs * n_steps, n_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + n_steps]
    
    n_batches = n_subseqs // batch_size
    for i in range(0, batch_size * n_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, n_steps):
    offset = random.randint(0, n_steps)
    n_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + n_tokens]).reshape(batch_size, -1)
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + n_tokens]).reshape(batch_size, -1)
    n_batchs = Xs.shape[1] // n_steps
    for i in range(0, n_steps * n_batchs, n_steps):
        X = Xs[:, i: i + n_steps]
        Y = Ys[:, i: i + n_steps]
        yield X, Y
    
class SeqDataLoader:
    def __init__(self, batch_size, n_steps, use_random_iter, max_tokens, path='E:/Datasets/timemachine/timemachine.txt'):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        
        self.corpus, self.vocab = load_corpus_time_machine(path=path, max_tokens=max_tokens)
        self.batch_size, self.n_steps = batch_size, n_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.n_steps)

def load_data_time_machine(batch_size, n_steps, use_random_iter=False, max_tokens=10000, path='E:/Datasets/timemachine/timemachine.txt'):
    data_iter = SeqDataLoader(batch_size, n_steps, use_random_iter, max_tokens, path)
    return data_iter, data_iter.vocab

def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(1, device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor(outputs[-1], device=device).reshape((1, 1))

    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for p in params:
            p.grad[:] *= theta / norm



## Language Model

def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state = None
    time_start, loss_sum, m = time.time(), 0.0, 0

    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)

        loss_sum += l.item()
        m += 1
    
    return math.exp(loss_sum / m), m * X.shape[0] * X.shape[1] / (time.time() - time_start)
        
def train_lm(net ,train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    net = net.to(device)
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)

    pred = lambda prefix : predict(prefix, 50, net, vocab, device)

    ppls = []
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        ppls.append(ppl)

        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1:d}, perplexity {ppl:.1f}')
            print(pred('time traveller'))
        
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.grid(True)
    ax.set_xlabel('epoch')
    ax.set_ylabel('perplexity')
    ax.plot(range(1, num_epochs + 1), ppls)
    plt.show()
    
    print(f'perplexity {ppl:.1f}, {speed:.1f} examples/sec on {str(device)}')
    print(pred('time traveller'))
    print(pred('traveller'))



## Sequence-to-Sequence

def read_data_nmt(path='E:/Datasets/Tatoeba-fra-eng/fra-eng/fra.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]

    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break

        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    
    return source, target

def truncate_pad(line, num_steps, padding_token):
    if(len(line) > num_steps):
        line = line[: num_steps]
    else:
        line += [padding_token] * (num_steps - len(line))
    
    return line

def build_arr_nmt(lines, vocab, num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(dim=1)

    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_arr_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_arr_nmt (target, tgt_vocab, num_steps)

    dataset = torch.utils.data.TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_iter, src_vocab, tgt_vocab

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
    
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)

        return self.decoder(dec_X, dec_state)

class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
    
    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        return output, state
    
class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)

        return output, state

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        
        return weighted_loss
    
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    losses = []
    start_time, tokens_sum = time.time(), 0
    for epoch in range(num_epochs):
        loss_sum = 0.0
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, : -1]], 1) # 这样处理后，dec_input就可以和target在时间步上一一对应
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()
            with torch.no_grad():
                loss_sum += l.mean().item()
                tokens_sum += X_valid_len.sum()
            
        losses.append(loss_sum / len(data_iter))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1:d}, loss {losses[-1]:.4f}')
    
    time_sum = time.time() - start_time
        
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(True)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(range(1, num_epochs + 1), losses)
    plt.show()

    print(f'loss {losses[-1]:.4f}, {tokens_sum / time_sum:.1f} tokens/sec on {str(device)}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    net.to(device)
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()

        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))

    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_pred - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    
    return score



## Attention

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
    
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

    return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    # queries.shape: (batch_size, num_queries, query_size)
    # keys.shape: (batch_size, num_keys, key_size)
    # values.shape: (batch_size, num_keys, value_size)
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)

        # shape: (batch_size, num_keys, value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    # queries.shape: (batch_size, num_queries, d)
    # keys.shape: (batch_size, num_keys, d)
    # values.shape: (batch_size, num_keys, value_size)
    # valid_len.shape: (batch_size, ) or (batch_size, num_queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)

class AttentionDecoder(Decoder):
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
    
    @property
    def attention_weights(self):
        raise NotImplementedError

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    
    # queries.shape: (batch_size, num_queries, query_size)
    # keys.shape: (batch_size, num_keys, key_size)
    # values.shape: (batch_size, num_keys, value_size)
    # valid_lens.shape: (batch_size, ) or (batch_size, num_keys)
    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        
        # (batch_size * num_heads, num_queries, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # (batch_size, num_queries, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)

        return self.W_o(output_concat)

# (batch_size, num_queries, num_hiddens) => (batch_size * num_heads, num_queries, num_hiddens / num_heads)
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

# (batch_size * num_heads, num_queries, num_hiddens / num_heads) => (batch_size, num_queries, num_hiddens)
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class PositionalEncoding(nn.Module):
    # 确保num_hiddens是偶数，不然会出问题
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32).reshape(1, -1) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
    
    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        
        return self.dropout(X)



## Transformer

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = myd2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X , self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(myd2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = myd2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                'block' + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias)
            )
    
    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        
        return X

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = myd2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = myd2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device = X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)        
        Z2 = self.ffn(Z)
        return self.addnorm3(Z, Z2), state

class TransformerDecoder(myd2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encodings = myd2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                'block' + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_inputs, ffn_num_hiddens, num_heads, dropout, i)
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encodings(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights



## Optimization Algorithms

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.1)
    
    ax = myd2l.build_ax('x', 'f(x)', figsize=(4, 3))
    ax.plot(f_line, f(f_line))
    ax.plot(results, [f(x) for x in results], c='orange')
    ax.plot(results, [f(x) for x in results], c='orange', marker='o')

def show_trace_2d(results, f):
    (list_x1, list_x2) = zip(*results)
    ax = myd2l.build_ax('x1', 'x2', grid=False, figsize=(5, 4))
    ax.plot(list_x1, list_x2, '-o', c='orange')
    max_x1, max_x2 = max(abs(max(list_x1)), abs(min(list_x1))), max(abs(max(list_x2)), abs(min(list_x2)))
    x1, x2 = torch.meshgrid(torch.arange(-max_x1 * 1.1, max_x1 * 1.1, 0.1), torch.arange(-max_x2 * 1.1, max_x2 * 1.1, 0.1), indexing='ij')
    ax.contour(x1, x2, f(x1, x2), colors='#1f77b4')



## Word Embedding

def read_ptb(path='E:\Datasets\ptb\ptb.train.txt'):
    with open(path) as f:
        raw_txt = f.read()
    return [line.split() for line in raw_txt.split('\n')]

# 下采样，去掉部分出现次数非常多的词，这类词一般无意义
def subsample(sentences, vocab):
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = myd2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    def keep(token):
        return random.uniform(0, 1) < math.sqrt(1e-4 / (counter[token] / num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences], counter)

def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        if len(line) < 2:
            continue

        centers += line
        for i in range(len(line)):
            windows_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - windows_size), min(len(line), i + windows_size + 1)))
            indices.remove(i)
            contexts.append([line[i] for i in indices])

    return centers, contexts

class RandomGenerator:
    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candicates = []
        self.i = 0
    
    def draw(self):
        if self.i == len(self.candicates):
            self.candicates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candicates[self.i - 1]
    
def get_negatives(all_contexts, vocab, counter, K):
    # 幂0.75是为了降低高频词的权重，让低频词也有机会被负采样
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negtives = []
        while len(negtives) < len(contexts) * K:
            neg = generator.draw()
            if neg not in contexts:
                negtives.append(neg)
        all_negatives.append(negtives)
    
    return all_negatives

def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers.append(center)
        contexts_negatives.append(context + negative + [0] * (max_len - cur_len))
        masks.append([1] * cur_len + [0] * (max_len - cur_len))
        labels.append([1] * len(context) + [0] * (max_len - len(context)))
    
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(contexts_negatives), torch.tensor(masks), torch.tensor(labels))

def load_data_pth(batch_size, max_window_size, num_noise_words):
    sentences = read_ptb()
    vocab = myd2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negtives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
        
        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index], self.negatives[index])
        
        def __len__(self):
            return len(self.centers)
    
    dataset = PTBDataset(all_centers, all_contexts, all_negtives)
    # 在每次从 DataLoader 中取出数据时，collate_fn 函数会被调用
    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, collate_fn=batchify
    )

    return data_iter, vocab

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets, mask=None):
        # logits 是指应用 Sigmoid 函数之前的输入值
        out = nn.functional.binary_cross_entropy_with_logits(inputs, targets, weight=mask, reduction='none')
        return out.mean(dim=1)

class TokenEmbedding:
    def __init__(self, embedding_name):
        self.paths = dict()
        self.paths['glove.6b.50d'] = 'E:/Datasets/glove/glove.6B.50d/'
        self.paths['glove.6b.100d'] = 'E:/Datasets/glove/glove.6B.100d/'
        self.paths['glove.42b.300d'] = 'E:/Datasets/glove/glove.42B.300d/'
        self.paths['wiki.en'] = 'E/Datasets/fastText/wiki.en/'

        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
    
    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = self.paths[embedding_name]
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                # rstrip() 方法删除最后多余的空格, \t, \n
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec

        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        
        return vecs
    
    def __len__(self):
        return len(self.idx_to_token)