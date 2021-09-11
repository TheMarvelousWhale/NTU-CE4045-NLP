# System Architecture of Part 4 

_the greatest work for 10m because we decide what matters, not anyone else_

![Booba Sword](./assets/baal-genshin-impact.gif)

The question asks us to "Define and develop a simple NLP application based on the dataset.". And so by all means since it's a dataset about restaurant review and we are by no means interested in all of those classification tasks, what we did is:

## Restaurant Review Generator with (distil) GPT2

It is a distilgpt2 model finetuned on the reviews such that given a certain prompts about the restaurants, the model will generate a few reviews corresponding to input attributes. The whole work consists of:
* Model training on [google colab](https://colab.research.google.com/drive/1owQQxXHd5MCKf1Dp-mf2jEWii5EXjjvd?usp=sharing) 
* Experiments monitoring with [wandb](https://wandb.ai/groupx/CE4045-Review-Generator/)
* Data lineage webapp built on [streamlit](https://streamlit.io/)
* Deployment on [Cloud Run](https://cloud.google.com/run) in GCP 
* Endpoint as an android app (sorry iphone users)


## **1. Feature Engineering**

We encode the 4 attribute in the review dataset -- "useful","stars","funny","cool" as text features, and feed it into the model in the format 
```
\<BOS\> features \<SEP\>review \<EOS\>
```
This allows us to form a partial control over the output review as we hope that the model can capture the relationship between a review and its various attributes. 

## **2. Model building**

As we know when it's come to text generation task it's always easier to use GPT2 due to its causal LM training objective (and also the ease of finetuning by simply giving it labels==input_inds). However the gpt2 pretrain is big in size (600MB) and has long inference time. As such distil gpt-2 was chosen for the task due to **PRODUCTION** reason. (Of course, if you just want to score for the assignment you wouldnt need to care for this issue)

The evaluation loss for such decision is about 0.1, for which I couldnt give a single fk about. Not like we can evaluate the generation output properly anyway. 

Colab was used because our dataset is about 15000 (train+dev) reviews, which running a cpu would cost about 138 hours (depends on batchsize, padding and other parameters). We use wandb to track the hyperameters and data lineage, which were extremely useful to judge which change results in certain behaviours of our model (hint: the number of trainable layers were one of the most interesting). 

One caveat is that wandb doesnt accept large file, so for the model output we have to use one drive (could have used git-lf, but well...)

## **3. Deployment**

We wanted to use TFX for this project, for its pruning and optimization as well as TF Lite deployment, but turns out turning a pytorch code (huggingface transformers) into TF is a nightmare. We abandon the idea and go with a serverless solution instead. 

Cloud Run was chosen because of its feasibility and free $300 gcp credits. The rest is history

## **4. Android**

The android app is the user touchpoint to our work, and it helps use to collect the user input (which can be done via firebase) and monitor these production data to see what our app is being used for, from which necessary changes can be made to improve UX. 

## **5. Streamlit**

It is important to monitor data lineage and provenance -- where the data comes from, what are its attributes, how have it been processed for each model so that if we want to reproduce ANY experiments we can do so with ease. Streamlit helps use visualize and monitor these qualities via a simple build procedure, and all in python. 


# References for our work
1. [Ivan Lai's medium on training GPT-2 for text gen task](https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d)
2. [Deploying model on Cloud Run](https://huggingface.co/blog/how-to-deploy-a-pipeline-to-google-clouds)
