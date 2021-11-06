###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
import random

import data_fnn as data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./Experiment1.pt',
                    help='model (.pt file) to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='15',
                    help='number of words to generate')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")


model = torch.load(args.checkpoint).to(device)
corpus = data.Corpus(args.data)
full_corpus = torch.cat((corpus.train, corpus.valid, corpus.test))


model.eval()
with open(args.outf, 'w') as outf:
    seed_pos = random.randint(0, len(full_corpus)-7)
    seed_span = full_corpus[seed_pos:seed_pos+7] # Pick random span from corpus
    generated_text=seed_span.to(device)
    for i in range(args.words):
        with torch.no_grad():
            output = model(generated_text[-7:])
            word_id = torch.argmax(output, dim=1)
            generated_text = torch.cat((generated_text,word_id))

    sent = [corpus.dictionary.idx2word[i] for i in generated_text]
    output_sent = ' '.join(sent)
    output_sent = output_sent.replace("<eos>", "\n")
    outf.write(output_sent)
    sent = sent[:7] + ['|']+sent[7:]
    print("###",' '.join(sent))
    orig = [corpus.dictionary.idx2word[i] for i in full_corpus[seed_pos:seed_pos+len(sent)-7]] 
    print("$$$",' '.join(orig))
    
