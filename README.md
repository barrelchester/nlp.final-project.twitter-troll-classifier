# nlp.final-project.twitter-troll-classifier

Given 3 million known Russian troll tweets and an equal number of normal tweets, develope a classifier to distinguish them.


Folder structure:

app - final python modules and notebooks
data - place to put original data - russian troll tweets and normal tweets
dev - place to experiment and figure out code that will eventually be put in modules and notebooks in app folder


Main Components
- preprocessing - cleaning and standardizing the text
- vectorization - text to numeric form
- analysis - (can happen during preprocessing and vectorization) analyze the data to flag and discard anomalies, calculate feature importance, etc
- modelling - experiment with different binary classifier models


Each main component should accept a list that specifies which method(s) to use for that component so that ablation analysis can be done. 
For instance: 
preprocess(methods=['lowercase', 'lemmatize', 'extract_entities'])
vectorize(methods=['tfidf_vectorizer', 'lsa'])
modelling(methods=['svm'])


