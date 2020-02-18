# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
from string import punctuation
#from nltk.corpus import stopwords
#from nltk.tokenize import TweetTokenizer
import json
import nltk
from collections import Counter
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.snowball import SnowballStemmer
#from nltk.corpus import sentiwordnet as swn

punctuation2 = punctuation.replace("#","") 
punctuation2 += '´΄’…“”–—―»«'

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       "]+", flags=re.UNICODE)

def RemoveFoursquare(dataset):
    data4 = []
    notdata4 = []
    for item in dataset:
        if "Foursquare" in item['source']:
            data4.append(item)
        else:
            notdata4.append(item)    
    return (data4,notdata4)
            
            
def GetUniqueNumberListTweets(datastore):
    lid = []
    for item in datastore:
        lid.append(item['id'])
    n = set(lid)
    return n

def GetTweetsbyUser(datastore, uid):
    u_datastore= []
    for item in datastore:
        if item['userid']==uid:
            u_datastore.append(item)
    return u_datastore
    
def GetTweetsbyID(datastore,tid):
    for item in datastore:
        if int(item['id'])==tid:
            return item
            


def GetTweetsby(datastore, mod):
    tlist = []
    for item in datastore:
        tlist.append(item['%s'%mod])
  
    tcounts = Counter(tlist)
    
    # Separate tweets according to a tag   
    f_datastore = {}         
    for k in tcounts.keys():
        f_datastore[k]=[]
        for item in datastore:
            if item['%s'%mod] == k:
                f_datastore[item['%s'%mod]].append(item)
    return (tcounts,f_datastore)    


def tweet_clean(tweet):

    # Remove HTML special entities (e.g. &amp;)
    tweet_c1 = re.sub(r'\&\w*;', '', tweet)

    # Remove hyperlinks
    tweet_c2 = re.sub(r'https?:\/\/.*\/\w*', '', tweet_c1)

    # Remove punctuation    
    tweet_c3 = re.sub(r'[' + punctuation2.replace('@', '') + ']+', ' ', tweet_c2)
    
    # Conversion to lowercase
    tweet_c4 = tweet_c3.lower()
    
    # Remove emoticons
    tweet_c5 = emoji_pattern.sub(r'', tweet_c4)
    
    # Tokenize with WhitespaceTokenizer to handle hashtag
    tokens = WhitespaceTokenizer().tokenize(tweet_c5)
      
    # Remove stopwords 
    stop_words = set(nltk.corpus.stopwords.words('italian'))
    filt_words= [w for w in tokens if not w in stop_words]    
    
    # stemming words (...)
    stemmer = SnowballStemmer("italian")
    stem_words = [stemmer.stem(w) for w in filt_words]
    return stem_words
    
    
def tweet_clean_nostem(tweet):

    # Remove HTML special entities (e.g. &amp;)
    tweet_c1 = re.sub(r'\&\w*;', '', tweet)

    # Remove hyperlinks
    tweet_c2 = re.sub(r'https?:\/\/.*\/\w*', '', tweet_c1)

    # Remove punctuation    
    tweet_c3 = re.sub(r'[' + punctuation2.replace('@', '') + ']+', ' ', tweet_c2)
    
    # Conversion to lowercase
    tweet_c4 = tweet_c3.lower()
    
    # Remove emoticons
    tweet_c5 = emoji_pattern.sub(r'', tweet_c4)
    
    # Tokenize with WhitespaceTokenizer to handle hashtag
    tokens = WhitespaceTokenizer().tokenize(tweet_c5)
      
    # Remove stopwords 
    stop_words = set(nltk.corpus.stopwords.words('italian'))
    filt_words= [w for w in tokens if not w in stop_words]    
    
    return filt_words


filename = "/home/monia/Downloads/tweets.json/tweets.json"

#Read JSON data into the datastore variable: 91893 tweets
ds=[]
with open(filename,'r') as f:
    for line in f:
        ds.append(json.loads(line))


# Get twitters by language
(langcounts,lang_ds) = GetTweetsby(ds, 'lang')


""" work on ita twitter"""

# clean tweets
clean_ds = lang_ds['it']

for item in clean_ds:
    item['text'] = tweet_clean(item['text'])
    
 
# filter tweets according to keywords   
list_means = ['auto','treno','trenitalia','atm','atac','gtt','5t','eav','busitalia','trenord','bus','autobus','tram','metropolitana','metro','bicicletta','bici','skateboard','moto','motocicletta','camion','camper','suv']
list_infrastructure = ['ferrovia','autostrada','fermata','stazione','tramvia','parcheggio','tangenziale']
list_action = ['sciopero','ritardo','incidente']

lst_t = list_means+list_infrastructure+list_action
stemmer = SnowballStemmer("italian")
stem_lst_t = [stemmer.stem(w) for w in lst_t]


# Extract id of filtered tweets      
kwords_ls=[]
for item in clean_ds:
    for i in item['text']:
        if i in stem_lst_t:       
            kwords_ls.append(int(item['id']))


# Extract mobility related text tweets for training
#----- upload agaoin ds because it has been changed during cleaning
ds=[]
with open(filename,'r') as f:
    for line in f:
        ds.append(json.loads(line))

(langcounts,lang_ds) = GetTweetsby(ds, 'lang')
#----

kwords_ds=[]
for lid in list(set(kwords_ls)):
    t = GetTweetsbyID(lang_ds['it'],lid)
    kwords_ds.append(t)

    
f = open('/home/monia/'+'dict_out_help.csv', 'w')

for k in kwords_ds:
       clean = tweet_clean_nostem(k['text'])
       a = " ".join(clean)  
       b = a.encode('utf-8')
       f.write(b)
       f.write(",")    
       f.write("%s"%k['id'])
       f.write('\n')
f.close()

# After manual classification
filename = "/home/monia/Documents/dict_out_class.csv"

f = open(filename, "r")
list_sent_p = []
list_sent_n = []

for l in f.readlines():
    text = l.split(",")[0]
    sent = (l.split(",")[2])[:-1]
    if sent=='n':
        sent_end = 'negative'
        list_sent_n.append((text,sent_end))
    else:
        sent_end='positive'
        list_sent_p.append((text,sent_end))
        
        
tweets = []
for (words, sentiment) in list_sent_p + list_sent_n:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))
    
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
    
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
#----------------

all_tw = "/home/monia/Downloads/tweetsPrin.json"
ds_all=[]
with open(all_tw,'r') as f:
    for line in f:
        ds_all.append(json.loads(line))

(langcounts_all,lang_ds_all) = GetTweetsby(ds_all, 'lang')

# clean tweets
clean_ds_all = lang_ds_all['it']
for item in clean_ds_all:
    item['text'] = tweet_clean(item['text'])
 
# extract by keywords   
kwords_ds_all=[]
for item in clean_ds_all:
    for i in item['text']:
        if i in stem_lst_t:       
            kwords_ds_all.append(item)



# re-create original PRIN dataset 
ds_all=[]
with open(all_tw,'r') as f:
    for line in f:
        ds_all.append(json.loads(line))

(langcounts_all,lang_ds_all) = GetTweetsby(ds_all, 'lang') 
 
 
 
# create the list of keywords-based tweets without stemming editing
kwords_ds_all_final=[]
list_ok = GetUniqueNumberListTweets(kwords_ds_all)
for list_item in list_ok:
    t=GetTweetsbyID(ds_all,int(list_item))
    t['text'] = tweet_clean_nostem(t['text'])
    kwords_ds_all_final.append(t)
    



ds_all_negative=[]   
ds_all_positive=[]
for item in kwords_ds_all_final:
    if classifier.classify(extract_features(item['text'])) == 'negative':
        ds_all_negative.append(item)
    if classifier.classify(extract_features(item['text'])) == 'positive':
        ds_all_positive.append(item)
        
        

