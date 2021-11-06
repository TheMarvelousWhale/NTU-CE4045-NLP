# coding: utf-8
import argparse
import torch
import math
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import data_fnn as data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 FNN Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--hidd', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--contsz', type=float, default=7,
                    help='context size (8-gram -> contsz = 7)')
parser.add_argument('--epochs', type=int, default=8,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size')
parser.add_argument('--tied', action='store_true',
                    help='tie the embedding weights and output weights (when using tied, contsz must be equal to hidd')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')


args = parser.parse_args()
EMBEDDING_DIM = args.emsize
CONTEXT_SIZE = args.contsz
BATCH_SIZE = args.batch_size

# hidden units
H = args.hidd
torch.manual_seed(42)
learn_rate = args.lr
tied = True if args.tied else False


###############################################################################
# Helper Functions
###############################################################################
def ngram_split(orig_corpus, dataset, n):
    # This function breaks corpus into [context, target]
    # For e.g., in trigram, the tensor returned would be [C(n-2), C(n-1), T]
    ngram = []
    data_len = len(dataset)
    eos_id = corpus.dictionary.word2idx['<eos>']       
    for i, tokenid in enumerate(dataset):
        if i+n<data_len:
            temp_gram = dataset[i:i+n+1].view(-1)
            if eos_id in temp_gram[0:n]:
                continue
            ngram.append(temp_gram)
    fin_ngram=torch.stack(ngram)
    return fin_ngram

def get_accuracy_from_log_probs(log_probs, labels):
    probs = torch.exp(log_probs)
    predicted_label = torch.argmax(probs, dim=1)
    acc = (predicted_label == labels).float().mean()
    return acc


def evaluate(model, criterion, dataloader, gpu):
    # helper function to evaluate model on dev data
    model.eval()
    mean_acc, mean_loss = 0, 0
    count = 0
    with torch.no_grad():
        dev_st = time.time()
        for it, data_tensor in enumerate(dataloader):
            context_tensor = data_tensor[:,0:CONTEXT_SIZE]
            target_tensor = data_tensor[:,CONTEXT_SIZE]
            context_tensor, target_tensor = context_tensor.cuda(gpu), target_tensor.cuda(gpu)
            log_probs = model(context_tensor)
            mean_loss += criterion(log_probs, target_tensor).item()
            mean_acc += get_accuracy_from_log_probs(log_probs, target_tensor)
            count += 1
            if it % 500 == 0: 
                print("Dev Iteration {} complete. Mean Loss: {}; Mean Acc:{}; Time taken (s): {}".format(it, mean_loss / count, mean_acc / count, (time.time()-dev_st)))
                dev_st = time.time()

    return mean_acc / count, mean_loss / count

###############################################################################
# Setup
###############################################################################
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda" if args.cuda else "cpu")

print("Preparing corpus...")
corpus = data.Corpus(args.data)
train_set = corpus.train
val_set = corpus.valid
test_set = corpus.test
train_ngram = ngram_split(corpus, train_set, CONTEXT_SIZE)
val_ngram = ngram_split(corpus, val_set, CONTEXT_SIZE)
test_ngram = ngram_split(corpus, test_set, CONTEXT_SIZE)
train_loader = DataLoader(train_ngram, batch_size = BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(val_ngram, batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ngram, batch_size = BATCH_SIZE, shuffle=True)


vocab_len = len(corpus.dictionary)
# Using negative log-likelihood loss
loss_function = nn.NLLLoss()
# init model
model = model.FNNModel(vocab_len, EMBEDDING_DIM, CONTEXT_SIZE, H, tied)
# load it to gpu
model.cuda(device)

# using ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr = learn_rate)


###############################################################################
# Training code
###############################################################################

# ------------------------- TRAIN & SAVE MODEL ------------------------
best_ppl = 9999
best_model_path = None
for epoch in range(args.epochs):
    st = time.time()
    print("\n--- Training model Epoch: {} ---".format(epoch+1))
    for it, data_tensor in enumerate(train_loader):
        context_tensor = data_tensor[:,0:CONTEXT_SIZE]
        target_tensor = data_tensor[:,CONTEXT_SIZE]
#         print(context_tensor)
#         print(target_tensor)
        context_tensor, target_tensor = context_tensor.cuda(device), target_tensor.cuda(device)

        # zero out the gradients from the old instance
        model.zero_grad()

        # get log probabilities over next words
        log_probs = model(context_tensor)
        # calculate current accuracy
        acc = get_accuracy_from_log_probs(log_probs, target_tensor)

        # compute loss function
        loss = loss_function(log_probs, target_tensor)

        # backward pass and update gradient
        loss.backward()
        optimizer.step()

        if it % 500 == 0: 
#             print("Training Iteration {} of epoch {} complete. Loss: {}; Acc:{}; Time taken (s): {}".format(it, epoch, loss.item(), acc, (time.time()-st)))
            st = time.time()

    print("\n--- Evaluating model on dev data ---")
    dev_acc, dev_loss = evaluate(model, loss_function, dev_loader, device)
    ppl = math.exp(dev_loss)
    print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}; Perplexity: {}".format(epoch+1, dev_acc, dev_loss, ppl))
    if ppl < best_ppl:
        print("Best development accuracy improved from {} to {}, saving model...".format(best_ppl, ppl))
        best_ppl = ppl
        # set best model path
        best_model_path = 'best_model_{}_gram_{}_{}H.dat'.format(CONTEXT_SIZE+1, epoch, H)
        # saving best model
        torch.save(model.state_dict(), best_model_path)

print("\nTraining Done. Performing Test...")
model.load_state_dict(torch.load(best_model_path))
test_acc, test_loss = evaluate(model, loss_function, test_loader, device)
print("Test Perplexity:", math.exp(test_loss))
torch.save(model, args.save)
print("\nModel saved as {}".format(args.save))
print("Done.")