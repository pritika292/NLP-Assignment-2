import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import re
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.activation(self.W1(input_vector))

        # [to fill] obtain output layer representation
        output = self.W2(hidden)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)
        
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def data_analysis(data):
    with open(data) as data_file:
        data_set = json.load(data_file)

    tot_count = 0
    maxlen = 0
    minlen = float('inf')

    counts = [0]*5
    for elt in data_set:
        review_len = len(elt["text"].split())
        if review_len > maxlen:
            maxlen = review_len
        if review_len < minlen:
            minlen = review_len
        tot_count += review_len

        rating = int(elt["stars"]-1)
        if rating == 0: 
            counts[0] += 1
        if rating == 1:
            counts[1] +=1
        if rating == 2:
            counts[2] +=1
        if rating == 3:
            counts[3] +=1
        if rating == 4:
            counts[4] +=1

    print("Review count : ", len(data_set))
    print("Max review length : ", maxlen)
    print("Min review length: ", minlen)
    print("Avg review length : ", tot_count/len(data_set))
    for i in range(len(counts)):
        print("Number of reviews with rating ", i, " = ", counts[i])
    
def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    replacement_rules = {'“': '"', '”': '"', '’': "'", '--': ','}
    for elt in training:
        elt["text"] = elt["text"].lower()
        for symbol, replacement in replacement_rules.items():
            elt["text"] = elt["text"].replace(symbol, replacement)
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        elt["text"] = elt["text"].lower()
        for symbol, replacement in replacement_rules.items():
            elt["text"] = elt["text"].replace(symbol, replacement)
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data_json, valid_data_json = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    print("Training data : ")
    data_analysis(args.train_data)
    print("\nValidation data : ")
    data_analysis(args.val_data)

    vocab = make_vocab(train_data_json)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data_json, word2index)
    valid_data = convert_to_vector_representation(valid_data_json, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))
    losses = [] 
    accuracies = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 

        total_loss = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(total_loss)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                # if predicted_label != gold_label:
                #     print(valid_data_json[minibatch_index * minibatch_size + example_index])
                #     print("TRUE : "+str(gold_label)+" PREDICTED : "+ str(predicted_label))
        accuracies.append(correct / total)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

    # write out to results/test.out
    with open("results/ffnn_result.out", "w") as outfile :
        outfile.write("Validation accuracy : {}".format(correct / total))

# print(losses)
# print(accuracies)

# Learning curve
epochs = list(range(1, len(losses) + 1))
plt.figure(figsize=(10, 4))

# Plot the training loss
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, label='Training Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss by Epoch')
plt.grid(True)
plt.legend()
plt.savefig('ffnn_training_loss.png')

# Plot the validation accuracy on the same graph
plt.figure(figsize=(10, 4))
plt.plot(epochs, accuracies, label='Validation Accuracy', marker='o', color='green')
plt.ylim(0.4, 0.8)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy by Epoch')
plt.grid(True)
plt.legend()
plt.savefig('ffnn_validation_accuracy.png')


    