import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

#Load data

with open('data/anna2.txt', 'r') as f:
    text = f.read()
    
chars = tuple(set(text))    #tuples of unique characters
int2char = dict(enumerate(chars))   # {0: ' ', 1: 'r', 2: 'p', 3: 't', 4: 'a'}
char2int = {ch: ii for ii, ch in int2char.items()} # {' ': 0, 'r': 1, 'p': 2, 't': 3, 'a': 4, 'y': 5, 's'}
print(char2int)
print("=====================")
encoded = np.array([char2int[ch] for ch in text]) #[ 2  7  3 11 10  9]. Each element corresponds to the character in dictionary (converted from text)

def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    
    batch_size = n_seqs * n_steps
    n_batches = len(arr)//batch_size
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        
        # The features
        x = arr[:, n:n+n_steps]
        
        # The targets, shifted by one
        y = np.zeros_like(x)
        
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y  # x is the sequence, y is the label


# test_encode = np.array([x for x in range(30)])
# n_seqs = 3
# n_steps = 5
# batch_size =  n_seqs * n_steps   # n_steps * n_seq
# batch_size = int(batch_size)
# n_batches = len(test_encode) // batch_size
# test_encode = test_encode[:n_batches*batch_size]
# print(test_encode)
# test_encode = test_encode.reshape((n_seqs, -1))
# print(test_encode)
# print(test_encode.shape)

# for n in range(0, test_encode.shape[1], n_steps):
#     # The features
#     x = test_encode[:, n:n+n_steps]
        
#     # The targets, shifted by one
#     y = np.zeros_like(x)
#     try:
#             y[:, :-1], y[:, -1] = x[:, 1:], test_encode[:, n+n_steps]
#     except IndexError:
#             y[:, :-1], y[:, -1] = x[:, 1:], test_encode[:, 0]

#     print(y)


class charRNN():
    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        #inputs
        # creating charater dictionaries
        self.chars = tokens
        self.int2char = dict(numerate(self.chars))
        self.char2int = {ch: ii for ii, ch in slef.int2char.items()}

        ## define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)

        ## define a dropout layer to reduce overfitting
        self.dropout = nn.Dropout(drop_prob)

        ## define the final, fully-connected output layer
        self.fc = nn.Linear(in_features=n_hidden, out_features=len(self.chars))

        # init the weights
        self.init_weights()

def forward(self, x, hc):
    # get x, and the new hidden state (h, c) from the lstm
    x, (h,c) =  self.lstm(x, hc)
    
    # pass x through the dropout layer
    x = self.dropout(x)

    # stack up LSTM outputs using view
    x = x.view(x.size()[0]*size()[1], self.n_hidden)

    # return x and the hidden state(h,c)

    return x, (h, c)

def predict(self, char, h=None, cuda=False, top_k=None):
    if cuda:
        self.cuda()
    else:
        self.cpu()
    
    if h is None:
        h = self.init_hidden(1)
    
    # create numpy array for input

    x = np.array([[self.char2int[char]]])
    x = one_hot_encode(x, len(self.chars))

    inputs = torch.from_numpy(x)

     if cuda:
        inputs = inputs.cuda()

    h = tuple([each.data for each in h])
    out, h = self.forward(inputs, h)

    p = F.softmax(out, dim=1).data

    if cuda:
        p = p.cpu()
    
    # if none just grab all of them
    if top_k in None:
        top_ch = np.arrange(len(self.chars))
    
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy.squeeze()
         
    p = p.numpy().squeeze()

    char = np.random.choice(top_ch, p=p/p.sum())

    return self.int2char[char], h

def init_weights(self):
    initrange = 0.1

    #set bias tensor to all zeros
    self.fc.bias.data.fill_(0)
    self.fc.weight.data.uniform_(-1,1)

def init_hidden(self, n_seqs):
    weight = next(self.parameters()).data
    return (weight.new(self.n_layers, n_seqs, self.nn_hidden).zero_(), weight.new(self.n_layers, n_seqs, self.n_hidden, self.n_hidden).zero_())
