import os


class Config():
    def __init__(self):
        self.data_path = '../data'
        
        self.troll_tweet_path = '%s/russian-troll-tweets' % self.data_path
        self.user_tweet_path = '%s/twitter_cikm_2010' % self.data_path

        #extracted unprocessed text path, is list of 'tweet type' tab 'tweet text'
        #tweet types are NormalUser, and for trolls: RightTroll, LeftTroll, NewsFeed, HashtagGamer and Fearmonger
        self.troll_tweet_texts_path = '%s/troll_tweets.pkl.gz' % self.data_path
        self.user_tweet_texts_path = '%s/user_tweets.pkl.gz' % self.data_path
        
        #cleaned tweets: 
        self.troll_tweet_clean_path = '%s/troll_tweets_clean.pkl.gz' % self.data_path
        self.user_tweet_clean_path = '%s/user_tweets_clean.pkl.gz' % self.data_path
        
        #tagged tweets (options: tokens, lemma, POS, phrases, entities)
        self.troll_tweet_tagged_path = '%s/troll_tweets_tagged.pkl.gz' % self.data_path
        self.user_tweet_tagged_path = '%s/user_tweets_tagged.pkl.gz' % self.data_path
        
        #tweet features
        self.troll_features_path = '%s/troll_features.pkl.gz' % self.data_path
        self.user_features_path = '%s/user_features.pkl.gz' % self.data_path
        
        #data subset
        self.feature_x_path = '%s/1000000_features_x.pkl.gz' % self.data_path
        self.feature_y_path = '%s/1000000_features_y.pkl' % self.data_path
        