# nlp.final-project.twitter-troll-classifier

Given a large set of Russian troll tweets and an equal number of normal tweets, develope a classifier to distinguish them.


### Folder structure:

app - final python modules
data - place to put original data - russian troll tweets, normal tweets, and other file artifacts
dev - holds python notebooks for walking through the experiments


### Instructions:

#### Libraries used:
numpy, matplotlib, sklearn, gensim (3.8.3 which fse needs), torch, spacy, fse, emoji

Additionally spacy and nltk resources need to be downloaded:
!{sys.executable} -m spacy download en_core_web_sm
!{sys.executable} -m pip install nltk
nltk.download('words')
nltk.download('wordnet')


#### 1. dev/data_processing.ipynb
This notebook calls the module that processes, cleans, parses, the tweet data. Instead of running it
you can just download the end results from OneDrive:

- data/troll_features.pkl.gz - https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/Ee83g1YadmlHvK-L1tiyoIMBGhlIMc0prbyRvnl_ctfQ9A?e=VqN8fM 
- data/user_features.pkl.gz - https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/EX_Hgvh_SgdBvXIFTmoDGuEB7C1_GAaUKXjBn_JuXHIjzA?e=VAAYr4


For further experiments, download the balanced subset of troll and user features from:
- data/1000000_features_x.pkl.gz -  https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/EQ6EamjRQm1Ar5_d3vHPOVUBITBgbuL5HUzXoPRJfeYFeQ?e=wmdQgm
- data/1000000_features_y.pkl.gz - https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/ESNlscav851PsHTJolfwckUBkvB2S3QT4S1XQVWkIHMCoA?e=zVupGL


If you want to see a subset of the original data it can be downloaded here:

- Normal user tweets: https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/Edm-7-ODB_RPuJ33U4mIzPoBfnPJG6xUhXFVa1kI-oc8qA?e=JW45EV
- Troll tweets: https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/EVVpYAo3A_BIjK1cXTFe26wBsKIi0BawqGzzbneWED8Gvg?e=OffTJl


#### 2. dev/vectorization_scikit-vectorizer-exploration
Explores sklearn vectorizers, models, and find relevant words via model coefficients.


#### 3. dev/vectorization_distributed-embedding-exploration.ipynb
Explores word2vec, fasttext, doc2vec, fast sentence embedding, sum tf-idf weighted word vectors, etc.
T-SNE plots are made to visualize class overlap and clusters.

These vectors for t-SNE visualization are available for download:
- sum_tfidf_weighted_ft_lemma_128_0.npy - https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/EU259N1OuqZFmbZtlXXt_RsBqxLWIoufx0WbFg6JzQtcSg?e=NevFq8
- sum_tfidf_weighted_w2v_lemma_128_0.npy - https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/EVz0dZghATNBryxmhnHVBfcBH2fJU7oORJyRzwygTYp7og?e=bfPzBh


#### 4. dev/modelling_scikit-models-exploration.ipynb
Many variations tested with sklearn models for insights into the data.


#### 5. dev/modelling_custom-models-exploration.ipynb
The custom torch model which gets the highest accuracy. Includes ablation studies on the model to find feature relevance.

The best model file can be downloaded here:
- lin_model_bin.pt - https://northeastern-my.sharepoint.com/:u:/g/personal/collins_wi_northeastern_edu/ESPtIGwWpUFBt4F2E2FejyMBl2lT6ypdBlIhh3HnumiKRA?e=EBPway


