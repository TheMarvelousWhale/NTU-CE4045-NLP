# Eh dont say i never warn you i dont know whether this one correct anot ah you better ownself check
# FNN 8-Gram Language Model

## Environment Setup (with Conda)
* Python env: `conda create --name 4045asg2 python=3.9.7`
* Start env: `conda activate 4045asg2`
* Install packages: `pip install -r requirements.txt`

## Training
```bash 
python train.py --cuda --epochs 6                     # Train a FNN on Wikitext-2 with CUDA
python train.py --cuda --epochs 6 --tied --hidd 200   # Train a tied FNN on Wikitext-2 with CUDA
python train.py --cuda --epochs 20                    # Train a FNN on Wikitext-2 with CUDA for 20 epochs
```

The `train.py` script accepts the following arguments:
```bash
optional arguments:
  -h, --help       show this help message and exit
  --data DATA      location of the data corpus
  --emsize EMSIZE  size of word embeddings
  --hidd HIDD      number of hidden units per layer
  --contsz CONTSZ  context size (8-gram -> contsz = 7)
  --epochs EPOCHS  upper epoch limit
  --batch_size N   batch size
  --tied           tie the embedding weights and output weights
  --cuda           use CUDA
  --save SAVE      path to save the final model
  --lr LR          initial learning rate
```

## Generating
```bash
python generate_fnn.py --cuda --checkpoint Experiment3.pt     # Generate text using Experiment3.pt model
python generate_fnn.py --cuda --checkpoint --words 30         # Generate text that are 30 tokens long
```

The `generate_fnn.py` script accepts the following arguments:
```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --checkpoint CHECKPOINT
                        model checkpoint to use
  --outf OUTF           output file for generated text
  --words WORDS         number of words to generate
  --cuda                use CUDA
```
`generate_fnn.py` generates text by selecting N random tokens (where N refers to the context size) from the corpus, then predicts the subsequent words. After running the script, two lines would appear on the bash. An example is as shown: 
```
### like a child , with doubts cast | in the first year , and the british empire had been destroyed by the national
$$$ like a child , with doubts cast over the of his reasons for refusing the under
``` 
The first line (starting with ###) is the generated text from the model, with the prompt and the predicted text separated by '|', and the second line (starting with $$$) is the original text from the corpus.