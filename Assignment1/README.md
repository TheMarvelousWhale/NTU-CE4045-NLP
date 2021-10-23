# NTU-CE4045-NLP
Group repo for the CE4045 Group Project


## Setup with conda
* Python env: `conda create --name 4045asg1 python=3.9.6`
* Start env: `conda activate 4045asg1`
* Install packages: `pip install -r requirements.txt`

## Running of components
### Data Exploration
* With `4045asg1` environment activated, start Jupyter Notebook: `jupyter notebook`
* Run `Data Exploration-Final.ipynb`

### Android Application
* Install `reviews_20211023.apk` in `Android` folder on an Android device/emulator
* Run the `NLP` application.
* Input a prompt, and click on 'Send'. A request will be made to hosted Google Cloud Run.

### Streamlit Web Application
* With `4045asg1` environment activated, in `DataExplorationStreamlit` folder, run `main.py` using command `streamlit run main.py`
* Navigate the page on http://localhost:8501


## Source Files
* `Review Generator with GPT-2.ipynb` : Notebook for Review Generator Model Development. To be mounted on Google Colab with `reviewSelected100.json` in Google Drive. 
* `/Gcloud_deploy` : Docker files for hosting of Review Generation Model
* `/Android` : Source code for Android App


## Pre-Processed/Pre-Trained Files
* `business_adj_phrase.json` : All Adjective Phrases, grouped by business
* `indicative_phrase-ltn` : Selected Indicative Adjective Phrase of each business (`ltn` refers to the SMART notation for tf-idf, i.e. log tf, idf, no normalisation)
* `POS_Tag.csv` : Output of the POS comparison between coarse grained and fine grained tagger
* `model/pretty-lion-15.pt` : Pre-trained Review Generator Model