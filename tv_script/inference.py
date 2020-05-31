import helper
data_dir = './data/Seinfeld_Scripts.txt'
from string import punctuation
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into a list of words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    print("Entering create_lookup_tables: input text:", text[:20])

    ##########
    # remove all punctuation (handles input as a list of words, possibly with punctuation,
    # including wrapping quotes (single or double))
    #.....turns out don't need this section since punctuation will be removed prior to calling this func
    
    #words_only = ' '.join([c for c in text if c not in punctuation])
    #words_only = ''.join([c for c in words_only if c not in punctuation]) #2nd pass to get embedded quotes
    
    ##########
    # create a set of words, forcing the contents to be the unique words in all scripts
    
    unique_words = Counter()
    unique_words.update(text)
    #print(len(unique_words), " unique words. Most common words:\n", unique_words.most_common(10))
    
    ##########
    # create the dictionary of words to ints (0-based indexing) and its inverse
    
    vocab_to_int = {}
    i = 0
    for w, c in unique_words.most_common(len(unique_words)):
        vocab_to_int[w] = i
        #if i < 60:
        #    print(w, ": ", vocab_to_int[w])
        i += 1
        
    int_to_vocab = {}
    #print()
    for w in vocab_to_int:
        index = vocab_to_int[w]
        int_to_vocab[index] = w
        #if index < 10:
            #print(index, ": ", w)
    
    #print("vocab_to_int size = ", len(vocab_to_int), ", int_to_vocab size = ", len(int_to_vocab))
    
    # return tuple
    return (vocab_to_int, int_to_vocab)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    ##########
    # need to manually assign everything
    #print(punctuation)
    d = {
        '<': "<LESSTHAN>",    #these 2 need to be first since we use the to delimit everything. When punctuation
        '>': "<GREATERTHAN>", #is substituted for these tokens the symbols will be done in this order.
        '-': "<HYPHEN>",       #this needs to be next since it is used inside several others
        '\n': "<RETURN>",
        '!': "<EXCLAMATION>",
        '"': "<DOUBLE-QUOTE>",
        '#': "<POUND>",
        '$': "<DOLLAR>",
        '%': "<PERCENT>",
        '&': "<AMPERSAND>",
        "'": "<SINGLE-QUOTE>",
        '(': "<LEFT-PAREN>",
        ')': "<RIGHT-PAREN>",
        '*': "<STAR>",
        '+': "<PLUS>",
        ',': "<COMMA>",
        '.': "<PERIOD>",
        '/': "<SLASH>",
        ':': "<COLON>",
        ';': "<SEMICOLON>",
        '=': "<EQUALS>",
        '?': "<QUESTION>",
        '@': "<AT>",
        '[': "<LEFT-BRACKET>",
        '\\': "<BACKSLASH>",
        ']': "<RIGHT-BRACKET>",
        '^': "<CARET>",
        '_': "<UNDERSCORE>",
        '`': "<TICK>",
        '{': "<LEFT-BRACE>",
        '|': "<PIPE>",
        '}': "<RIGHT-BRACE>",
        '~': "<TILDE>",
        '\t': "<TAB>"
    }
    #print(d)
        
    return d


import os
import pickle
import torch

SPECIAL_WORDS = {'PADDING': '<PAD>'}


# # Check Point
# This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')


# ## Input
# Let's start with the preprocessed input data. We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.
# 
# You can create data with TensorDataset by passing in feature and target tensors. Then create a DataLoader as usual.
# ```
# data = TensorDataset(feature_tensors, target_tensors)
# data_loader = torch.utils.data.DataLoader(data, 
#                                           batch_size=batch_size)
# ```
# 
# ### Batching
# Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.
# 
# >You can batch words using the DataLoader, but it will be up to you to create `feature_tensors` and `target_tensors` of the correct size and content for a given `sequence_length`.
# 
# For example, say we have these as input:
# ```
# words = [1, 2, 3, 4, 5, 6, 7]
# sequence_length = 4
# ```
# 
# Your first `feature_tensor` should contain the values:
# ```
# [1, 2, 3, 4]
# ```
# And the corresponding `target_tensor` should just be the next "word"/tokenized word value:
# ```
# 5
# ```
# This should continue with the second `feature_tensor`, `target_tensor` being:
# ```
# [2, 3, 4, 5]  # features
# 6             # target
# ```

# In[9]:


from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    ##########
    # iterate through all the words to construct features & targets based on sequence length
    
    total_words = len(words)
    end = total_words - sequence_length - 1 #account for the target word at end
    fl = []
    tl = []
    for i, w in enumerate(words):
        #print("i = ", i, ", w =", w, end=' ')
        if i > end:
            break
        fl.append(words[i:i+sequence_length])
        tl.append(words[i+sequence_length])
        #print("fl = ", fl, ", tl = ", tl)
        
    features = torch.from_numpy(np.array(fl))
    targets = torch.from_numpy(np.array(tl))
    #print("features = ", features, "\ntargets = ", targets)
    
    ##########
    # batch up the data
    
    data = TensorDataset(features, targets)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader



# ### Test your dataloader 
# 
# You'll have to modify this code to test a batching function, but it should look fairly similar.
# 
# Below, we're generating some test text data and defining a dataloader using the function you defined, above. Then, we are getting some sample batch of inputs `sample_x` and targets `sample_y` from our dataloader.
# 
# Your code should return something like the following (likely in a different order, if you shuffled your data):
# 
# ```
# torch.Size([10, 5])
# tensor([[ 28,  29,  30,  31,  32],
#         [ 21,  22,  23,  24,  25],
#         [ 17,  18,  19,  20,  21],
#         [ 34,  35,  36,  37,  38],
#         [ 11,  12,  13,  14,  15],
#         [ 23,  24,  25,  26,  27],
#         [  6,   7,   8,   9,  10],
#         [ 38,  39,  40,  41,  42],
#         [ 25,  26,  27,  28,  29],
#         [  7,   8,   9,  10,  11]])
# 
# torch.Size([10])
# tensor([ 33,  26,  22,  39,  16,  28,  11,  43,  30,  12])
# ```
# 
# ### Sizes
# Your sample_x should be of size `(batch_size, sequence_length)` or (10, 5) in this case and sample_y should just have one dimension: batch_size (10). 
# 
# ### Values
# 
# You should also notice that the targets, sample_y, are the *next* value in the ordered test_text data. So, for an input sequence `[ 28,  29,  30,  31,  32]` that ends with the value `32`, the corresponding output should be `33`.



# ---
# ## Build the Neural Network
# Implement an RNN using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module). You may choose to use a GRU or an LSTM. To complete the RNN, you'll have to implement the following functions for the class:
#  - `__init__` - The initialize function. 
#  - `init_hidden` - The initialization function for an LSTM/GRU hidden state
#  - `forward` - Forward propagation function.
#  
# The initialize function should create the layers of the neural network and save them to the class. The forward propagation function will use these layers to run forward propagation and generate an output and a hidden state.
# 
# **The output of this model should be the *last* batch of word scores** after a complete sequence has been processed. That is, for each input sequence of words, we only want to output the word scores for a single, most likely, next word.
# 
# ### Hints
# 
# 1. Make sure to stack the outputs of the lstm to pass to your fully-connected layer, you can do this with `lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)`
# 2. You can get the last batch of word scores by shaping the output of the final, fully-connected layer like so:
# 
# ```
# # reshape into (batch_size, seq_length, output_size)
# output = output.view(batch_size, -1, self.output_size)
# # get last batch
# out = output[:, -1]
# ```

# In[11]:


import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc0 = nn.Linear(hidden_dim, output_size)
    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        batch_size = nn_input.shape[0]

        x = self.emb(nn_input)
        x, hidden = self.lstm(x, hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.drop(x)
        x = self.fc0(x) #no activation
        #print("After fc: x.shape = ", x.shape)
        
        #reshape to have result as [batch_size, output]
        x = x.view(batch_size, -1, self.output_size)
        #print("After reshaping, x.shape = ", x.shape)
        #print("x[0] = ", x[0])
        rtn = x[:, -1]
        #print("Returning rtn.shape = ", rtn.shape)
        #print("x[0] = ", rtn[0])
        
        # return one batch of output words and the hidden state
        return rtn, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        weights = next(self.parameters()).data
        
        h = weights.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        c = weights.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        if train_on_gpu:
            h = h.cuda()
            c = c.cuda()
            
        hidden = (h, c)
        
        return hidden



# ### Define forward and backpropagation
# 
# Use the RNN class you implemented to apply forward and back propagation. This function will be called, iteratively, in the training loop as follows:
# ```
# loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)
# ```
# 
# And it should return the average loss over a batch and the hidden state returned by a call to `RNN(inp, hidden)`. Recall that you can get this loss by computing it, as usual, and calling `loss.item()`.
# 
# **If a GPU is available, you should move your data to that GPU device, here.**

# In[12]:


def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param rnn: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :param hidden: The hidden state from previous iteration
    :return: The loss and the latest hidden state Tensor
    """
    CLIP_LIMIT = 5.0
    
    if train_on_gpu:
        inp = inp.cuda()
        target = target.cuda()
    
    ##########
    # perform backpropagation and optimization
    
    #reset backprop data
    hidden = tuple([i.data for i in hidden])
    rnn.zero_grad()
    
    #run the rnn
    output, hidden = rnn(inp, hidden)
    #print("After running rnn: output = ", output.shape, ", hidden0 = ", hidden[0].shape, ", hidden1 = ", hidden[1].shape)
    
    #compute the loss
    #print("target = ", target)
    loss = criterion(output, target)
    #print("loss = ", loss, ", loss.item = ", loss.item())
    
    #backprop with gradient clipping
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), CLIP_LIMIT)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden




# ### Hyperparameters
# 
# Set and train the neural network with the following parameters:
# - Set `sequence_length` to the length of a sequence.
# - Set `batch_size` to the batch size.
# - Set `num_epochs` to the number of epochs to train for.
# - Set `learning_rate` to the learning rate for an Adam optimizer.
# - Set `vocab_size` to the number of uniqe tokens in our vocabulary.
# - Set `output_size` to the desired size of the output.
# - Set `embedding_dim` to the embedding dimension; smaller than the vocab_size.
# - Set `hidden_dim` to the hidden dimension of your RNN.
# - Set `n_layers` to the number of layers/cells in your RNN.
# - Set `show_every_n_batches` to the number of batches at which the neural network should print progress.
# 
# If the network isn't getting the desired results, tweak these parameters and/or the layers in the `RNN` class.


# Data params
# Sequence Length
sequence_length = 50  # of words in a sequence; was 5; submitted with 20
# Batch Size
batch_size = 256

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)


# Training parameters
# Number of Epochs
num_epochs = 10
# Learning Rate
learning_rate = 0.001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
print("vocab_size = ", vocab_size)
# Output size
#jas - I believe this needs to be equal to vocab_size so that each column of the output matrix represents
#      one of the possible words, and a row of this matrix is essentially trying to immitate a one-hot vector.
#      However, the forward() method only returns the final column of the output, which, if this is true, would
#      represent the final word in the vocabulary. Therefore, this doesn't seem right, but it is at least
#      avoiding runtime errors for now.
output_size = vocab_size #num words it will generate for each pass?
# Embedding Dimension
embedding_dim = 400
# Hidden Dimension
hidden_dim = 512
# Number of RNN Layers
n_layers = 3

# Show stats for every n number of batches
show_every_n_batches = 1000

print("Torch version: ", torch.__version__)
if train_on_gpu:
    print("///// Using GPU!")
else:
    print("///// limited to using cpu")



'''
# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
#with active_session():
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')
'''



"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn')


# ## Generate TV Script
# With the network trained and saved, you'll use it to generate a new, "fake" Seinfeld TV script in this section.
# 
# ### Generate Text
# To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. You'll be using the `generate` function to do this. It takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. Also note that it uses topk sampling to introduce some randomness in choosing the most likely next word, given an output set of word scores!



import torch.nn.functional as F

def generate(rnn, prime_ids, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param rnn: The PyTorch Module that holds the trained neural network
    :param prime_ids: The word ids to start the first prediction (a list)
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    #print("1. current_seq = ", current_seq)
    num_primes = len(prime_ids)
    #current_seq[-1][-num_primes] = prime_ids
    seq_size = sequence_length
    print("seq_size = ", seq_size, ", num_primes = ", num_primes)
    for i in range(num_primes):
        current_seq[-1][seq_size - num_primes + i] = prime_ids[i]
    print("2. current_seq = ", current_seq)
    #predicted = [int_to_vocab[prime_ids]]
    predicted = []
    for i in range(num_primes):
        predicted.append(int_to_vocab[prime_ids[i]])
    print("Predicted = ", predicted)
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        #print("\nTop of loop.")
        #print("p     = ", p)
        #print("top_i = ", top_i)
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        #print("word_i = ", word_i, ", word = ", word)
        #print("New predicted = ", predicted)
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = current_seq.cpu()
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences


# ### Generate a New Script
# It's time to generate the text. Set `gen_length` to the length of TV script you want to generate and set `prime_word` to one of the following to start the prediction:
# - "jerry"
# - "elaine"
# - "george"
# - "kramer"
# 
# You can set the prime word to _any word_ in our dictionary, but it's best to start with a name for generating a TV script. (You can also start with any other names you find in the original text file!)

# run the cell multiple times to get different results!
gen_length = 400 # modify the length to your preference
prime_word = 'newman' # name for starting the script
prime_words = ['jerry', 'hello', 'newman']
prime_ints = []
for w in prime_words:
    prime_ints.append(vocab_to_int[w])
print("Prime ints = ", prime_ints)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
pad_word = helper.SPECIAL_WORDS['PADDING']
#generated_script = generate(trained_rnn, vocab_to_int[prime_word], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
generated_script = generate(trained_rnn, prime_ints, int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
print(generated_script)


# save script to a text file
f =  open("generated_script_1.txt","w")
f.write(generated_script)
f.close()

