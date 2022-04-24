import os, re, gzip, pickle, csv

#these have to be installed first (see preproc notebook)
import spacy
from emoji import UNICODE_EMOJI
from nltk.corpus import words
from nltk.corpus import wordnet 



class Preprocessing():
    '''The first module to run. Prepares data for vectorization. Config contains all paths. 
    Order of operations and resulting files:
    - parse raw data and store tab delim tweet_type, tweet_text
    - clean raw parsed data and store tab delim tweet_type, tweet_text
    - NLP tag clean data using spacy, store tab delim tweet_type, tweet_text, tokens, lemmas, POS, phrases, entities'''
    
    def __init__(self, config):
        '''Config contains all paths and static variables'''
        self.config = config
        self.replacements = {'‘':"'", '’':"'", '“':'"', '”':'"'}
        self.special_tags = set(['<EMOJI>', '<LINK>', '<USER>'])
        self.punct = set(["'", '"', '.', ',', '~', '!', '@', '#', '$', '%', '^', '&', '*', '|',
             '(', ')', '-', '_', '+', '=', '{','}','[',']',';', ':', '<', '>', '?', '/'])
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.vocab = set(words.words()) | set(wordnet.words())
        
        #tell spacy not to tokenize these tags
        for tag in self.special_tags:
            self.spacy_nlp.tokenizer.add_special_case(f"%s" % tag, [{spacy.attrs.ORTH: f"%s" % tag}])
        
        
    def preprocess_data(self):
        '''Options is for ablation, vectorize & train models with/without certain preprocessing steps
        to see how they affect the outcome.
        
        options:
        use_link_user_emoji_tags - option to replace link/user/emoji with tags <LINK> etc.
        '''
        #first extract the tweet type and text from the raw files and store in compressed single files
        if not os.path.exists(self.config.troll_tweet_texts_path):
            print('%s not found, extracting tweets...' % self.config.troll_tweet_texts_path)
            self.extract_type_and_text()
            print('Done extracting and storing raw tweets.')
        else:
            print('Raw tweets already extracted.')
            
        #clean and normalize - replace link/user/emoji with tags <LINK> etc
        if not os.path.exists(self.config.troll_tweet_clean_path):
            print('%s not found, cleaning tweets...' % self.config.troll_tweet_clean_path)
            self.clean_tweets()
            print('Done cleaning and storing tweets.')
        else:
            print('Extracted tweets already cleaned.')
            
        #apply NLP tagging with spacy (tokens, lemma, POS, phrases, entities)
        if not os.path.exists(self.config.troll_tweet_tagged_path):
            print('%s not found, applying spacy NLP tagging to tweets (this may take a while)...' % self.config.troll_tweet_tagged_path)
            self.tokenize_tweets()
            print('Done tagging and storing tweets.')
        else:
            print('Clean tweets already tagged.')
            
        #calculate features and create final tweet feature records
        if not os.path.exists(self.config.troll_feature_path):
            print('%s not found, creating feature files' % self.config.troll_feature_path)
            self.get_features()
            print('Done tagging and storing tweets.')
        else:
            print('Tweet features already created.')
            
        print('All preprocessing tasks complete')
        
            
    def extract_type_and_text(self):
        '''Extract tweet_type tab tweet_text from raw troll and user tweets and store in compressed file'''
        troll_tweets = self.__extract_troll_tweets()
        
        with gzip.open(self.config.troll_tweet_texts_path, 'wb') as oz:
            pickle.dump(troll_tweets, oz)
            
        user_tweets = self.__extract_user_tweets()
        
        with gzip.open(self.config.user_tweet_texts_path, 'wb') as oz:
            pickle.dump(user_tweets, oz)
        
        
    def __extract_troll_tweets(self):
        '''Parses raw troll tweets. Stores file of 
        tweet_type (RightTroll, LeftTroll, NewsFeed, HashtagGamer and Fearmonger) tab tweet_text list.'''
        troll_tweets = []
        badtab=0
        badline=0
        nonenglish=0
        
        print('Extracting raw troll tweets...')
        
        for fn in os.listdir(self.config.troll_tweet_path):
            if not fn.endswith('.csv'):
                continue
                
            with open('%s/%s' % (self.config.troll_tweet_path, fn), 'r', encoding='utf-8', newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                
                header = next(reader)
                
                # Headers:
                #external_author_id, author, content, region, language, publish_date, arvested_date, following,
                #followers, updates, post_type, account_type, retweet, account_category, new_june_2018, alt_external_id,
                # tweet_id, article_url, tco1_step1, tco2_step1, tco3_step1
                
                for fields in reader:
                    #ensure correct parsing
                    if not len(fields)==21:
                        badtab+=1
                        continue

                    #omit non-english
                    if not fields[4].lower().strip()=='english':
                        nonenglish+=1
                        continue

                    #ensure presence of text
                    text = fields[2].strip()
                    if not text:
                        badline+=1
                        continue

                    #get troll type: RightTroll, LeftTroll, HashtagGamer, NewsFeed, FearMonger
                    troll_type = fields[13].lower().strip()
                    
                    troll_tweets.append('%s\t%s' % (troll_type, text))

        print('Extracted %d troll tweets (badtabs: %d, badlines: %d, skipped nonenglish: %d\nExample: %s\n)' % (
            len(troll_tweets), badtab, badline, nonenglish, troll_tweets[-1]))
        
        return troll_tweets


    def __extract_user_tweets(self):
        '''Parses raw user tweets. Stores file of tweet_type ("NormalUser") tab tweet_text list.'''
        user_tweets = []
        badtab=0
        badline=0
        
        print('Extracting raw user tweets...')

        for fn in os.listdir(self.config.user_tweet_path):
            if not fn.endswith('tweets.txt'):
                continue
            
            fp = '%s/%s' % (self.config.user_tweet_path, fn)
            with open(fp, 'r', encoding='utf-8') as f:
                for i,line in enumerate(f):
                    if i%1000000==0:
                        print(i)
                        
                    if not line.count('\t')==3:
                        badtab+=1
                        continue
                        
                    text = line.replace('\n','').split('\t')[2].strip()
                    if not text:
                        badline+=1
                        continue
                        
                    user_tweets.append('NormalUser\t%s' % text)

        print('bad tabs: %d, bad lines: %d' % (badtab, badline))
        print('Extracted %d normal user tweets (badtabs: %d, badlines: %d)\nExample: %s\n' % (
            len(user_tweets), badtab, badline, user_tweets[-1]))

        return user_tweets
    
    
    def clean_tweets(self):
        '''Reads raw tweet files and cleans the text, normalizing some chars, spaces, and replacing
        emojis, links, and user mentions with tags.'''
        #clean troll tweets
        with gzip.open(self.config.troll_tweet_texts_path, 'rb') as fz:
            troll_tweets = pickle.load(fz)
            
        print('Cleaning %d extracted troll tweets...' % len(troll_tweets))
        clean_troll_tweets = []
        for troll_tweet in troll_tweets:
            clean_troll_tweets.append(self.__clean_tweet(troll_tweet))
            
        print('Saving %d cleaned troll tweets to %s' % (len(clean_troll_tweets), self.config.troll_tweet_clean_path))
        with gzip.open(self.config.troll_tweet_clean_path, 'wb') as oz:
            pickle.dump(clean_troll_tweets, oz)
            
        #clean user tweets
        with gzip.open(self.config.user_tweet_texts_path, 'rb') as fz:
            user_tweets = pickle.load(fz)
            
        print('Cleaning %d extracted user tweets...' % len(user_tweets))
        clean_user_tweets = []    
        for i,user_tweet in enumerate(user_tweets):
            if i%1000000==0:
                print(i)
            clean_user_tweets.append(self.__clean_tweet(user_tweet))
            
        print('Saving %d cleaned user tweets to %s' % (len(clean_user_tweets), self.config.user_tweet_clean_path))
        with gzip.open(self.config.user_tweet_clean_path, 'wb') as oz:
            pickle.dump(clean_user_tweets, oz)


    def __clean_tweet(self, tweet):
        '''Applies char replacements to normalize quotes etc, normalizes whitespace, 
        replaces emojis, links, and user mentions with tags <EMOJI>, <LINK>, <USER>.'''
        #normalize spaces, quotes, double quotes, etc ‘’“”
        for f,r in self.replacements.items():
            tweet = tweet.replace(f,r)

        #norm spaces
        tweet = re.sub('  +', ' ', tweet)

        #replace links with <LINK>, replace usertags with <USER>, leave hashtags
        tweet = re.sub('https?:[^ ]+', '<LINK>', tweet)

        tweet = re.sub('@[a-zA-Z][^ ]+', '<USER>', tweet)

        clean = []
        for char in tweet:
            if char in UNICODE_EMOJI['en']:
                clean.append('<EMOJI>')
            else:
                clean.append(char)
        tweet = ''.join(clean)

        return tweet
    
    
    def tokenize_tweets(self):
        '''Loads cleaned tweets and runs them through spacy NLP pipeline to create tagged text files.'''
        #process troll tweets
        with gzip.open(self.config.troll_tweet_clean_path, 'rb') as fz:
            troll_tweets = pickle.load(fz)
        
        print('Tagging %d cleaned troll tweets...' % len(troll_tweets))
        tagged_troll_tweets = []
        for i,troll_tweet in enumerate(troll_tweets):
            tweet_type, tweet_text = troll_tweet.split('\t')
            toks, lemmas, pos, phrases, ents = self.tokenize_text(tweet_text)
            tagged_troll_tweets.append('%s\t%s\t%s\t%s\t%s\t%s\t%s' % (tweet_type, tweet_text, toks, lemmas, pos, phrases, ents))

            if i%100000==0:
                print('Storing troll tweets tagged so far:', i, tagged_troll_tweets[-1])
                with gzip.open(self.config.troll_tweet_tagged_path, 'wb') as oz:
                    pickle.dump(tagged_troll_tweets, oz)
                    
        print('Storing complete %d tagged troll tweets' % len(tagged_troll_tweets))
        with gzip.open(self.config.troll_tweet_tagged_path, 'wb') as oz:
            pickle.dump(tagged_troll_tweets, oz)
            
            
        #process user tweets
        with gzip.open(self.config.user_tweet_clean_path, 'rb') as fz:
            user_tweets = pickle.load(fz)
            
        print('Tagging %d cleaned troll tweets...' % len(troll_tweets))
        tagged_user_tweets = []
        for i,user_tweet in enumerate(user_tweets):
            tweet_type, tweet_text = user_tweet.split('\t')
            toks, lemmas, pos, phrases, ents = self.tokenize_text(tweet_text)
            tagged_user_tweets.append('%s\t%s\t%s\t%s\t%s\t%s\t%s' % (tweet_type, tweet_text, toks, lemmas, pos, phrases, ents))

            if i%100000==0:
                print('Storing user tweets tagged so far:', i, tagged_user_tweets[-1])
                with gzip.open(self.config.user_tweet_tagged_path, 'wb') as oz:
                    pickle.dump(tagged_user_tweets, oz)
                    
        print('Storing complete %d tagged user tweets' % len(tagged_user_tweets))
        with gzip.open(self.config.user_tweet_tagged_path, 'wb') as oz:
            pickle.dump(tagged_user_tweets, oz)


    def tokenize_text(self, text):
        '''Public method for running a tweet text through spacy NLP pipeline to get text, 
        tokens, lemmas, POS, phrases, and entities.'''
        toks = []
        lemmas = []
        pos = []
        phrases = []
        ents = []

        doc = self.spacy_nlp(text)
        for chunk in doc.noun_chunks:
            #only store multi word phrases
            if not ' ' in chunk.text:
                continue
                
            #join multiword phrases with _
            phrases.append(chunk.text.replace(' ', '_'))

        ent_type=''
        ent=[]
        for tok in doc:
            toks.append(tok.text)

            if tok.text in self.special_tags:
                lemmas.append(tok.text)
                pos.append('TAG')
                continue

            if tok.ent_iob == 3: #start
                ent_type = tok.ent_type_
                ent.append(tok.text)
            elif tok.ent_iob == 1: #continue
                ent.append(tok.text)
            else: 
                if ent: #done
                    #join multiword entities with _, like phrases
                    ents.append('%s:%s' % ('_'.join(ent), ent_type))
                    ent=[]
                    ent_type=''

            lemmas.append(tok.lemma_)
            pos.append(tok.tag_)

        toks = ' '.join(toks)
        lemmas = ' '.join(lemmas)
        pos = ' '.join(pos)
        phrases = ' '.join(phrases)
        ents = ' '.join(ents)

        return toks, lemmas, pos, phrases, ents
    
    
    def get_features(self):
        #process troll tweets
        with gzip.open(self.config.troll_tweet_tagged_path, 'rb') as fz:
            troll_tweets = pickle.load(fz)

        troll_feats = self.calculate_features(troll_tweets)

        print('storing troll features')
        with gzip.open(self.config.troll_feature_path, 'wb') as oz:
            pickle.dump(troll_feats, oz)

        #process user tweets
        with gzip.open(self.config.user_tweet_tagged_path, 'rb') as fz:
            user_tweets = pickle.load(fz)

        user_feats = self.calculate_features(user_tweets)

        print('storing user features')
        with gzip.open(self.config.user_feature_path, 'wb') as oz:
            pickle.dump(user_feats, oz)

        
    def calculate_features(self, tweets):
        feats = []
        for i,tweet in enumerate(tweets):
            if i and i%100000==0:
                print(i, feats[-1])

            tp,txt,toks,lems,pos,phrs,ents = tweet.replace('\xa0','').split('\t')

            if tp=='NonEnglish':
                continue

            num_toks = toks.count(' ')+1
            emoji_ratio, link_ratio, user_ratio = 0,0,0
            emoji_ratio = txt.count('<EMOJI>')/num_toks
            link_ratio = txt.count('<LINK>')/num_toks
            user_ratio = txt.count('<USER>')/num_toks

            toks = toks.replace('< ', '<').replace(' >', '>').replace('# ','#')

            clean_ents = set()
            ent_types = set()
            ent_toks = set()
            for ent in ents.split(' '):
                if ent=='#:CARDINAL':
                    continue
                items = ent.split(':')
                for t in items[0].split('_'):
                    ent_toks.add(t)
                typ = items[-1]
                if not typ:
                    continue
                if typ[0]=='#':
                    continue
                if '\xa0' in typ:
                    continue
                clean_ents.add(ent)
                ent_types.add(typ)
            clean_ents = list(clean_ents)
            ent_types = list(ent_types)
            ent_toks = list(ent_toks)

            num_ents = ents.count(' ')+1

            lems = lems.lower()
            lems = re.sub('<[^>]+>', '', lems)
            lems = lems.replace('# ','#')

            voc=[]
            novoc=[]
            for lem in lems.split(' '):
                if not lem or lem in self.punct or lem[0]=='#' or lem in ent_toks:
                    continue
                if lem in self.vocab:
                    voc.append(lem)
                else:
                    novoc.append(lem)

            ratio = 0
            if voc or novoc:
                ratio = len(novoc)/(len(voc)+len(novoc))
                #print('%s\nvoc: %s\nno voc: %s\nratio: %.6f\n' % (txt, voc, novoc, ratio))

            tags = re.findall('#[^ ]+', txt)

            feats.append({
                'type':tp,
                'text':txt,
                'tokens':toks,
                'lemmas':lems,
                'pos':pos,
                'phrases':phrs,
                'entities':clean_ents,
                'ent_types':ent_types,
                'hashtags':tags,
                'oov_words':' '.join(novoc),
                'emoji_ratio':emoji_ratio, 
                'link_ratio':link_ratio, 
                'user_ratio':user_ratio,
                'oov_ratio':ratio
            })

        return feats
    