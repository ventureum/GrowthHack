from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Phrases
from gensim import corpora, models
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from pprint import pprint
import nltk
np.random.seed(2018)
from sklearn.feature_extraction.text import TfidfVectorizer

from telethon import TelegramClient, sync, utils

# These example values won't work. You must get your own api_id and
# api_hash from https://my.telegram.org, under API Development.
api_id = 402623
api_hash = '64d49cbf562d45b5bf21c716a90a979f'


client = TelegramClient('session_name', api_id, api_hash)
client.start()

for dialog in client.get_dialogs(limit=10):
    print(dialog.name, dialog.draft.text)

data = []
for message in client.iter_messages('Cardstack ', limit=2000):
    if message.message is not None:
        data.append(message.message)
        # print(message.message)

print('{} number of messages loaded!'.format(len(data)))

stemmer = SnowballStemmer('english')


# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed# here I

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

'''
tfidf_vectorizer = TfidfVectorizer(max_df=0.99, max_features=200000,
                                 min_df=0.01, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(data)

print(tfidf_matrix.shape)
'''

sid = SentimentIntensityAnalyzer()

processed_data = []
for _d in data:
    processed_data.append(preprocess(_d))

bigram = Phrases(processed_data, min_count=5, threshold=3)

processed_data2 = []

dataMap = []
count = 0
for msg in processed_data:
    rv = bigram[msg]
    if len(rv) >= 10 and  len(rv) <= 30:
        processed_data2.append(rv)
        score = sid.polarity_scores(data[count])
        dataMap.append({'text':data[count], 'score': score})
    count+=1


dictionary = gensim.corpora.Dictionary(processed_data2)

dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)

bow_corpus  = [dictionary.doc2bow(doc) for doc in processed_data2]
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=5)

topics = lda_model_tfidf.print_topics(-1)

topic_docs = {}
topic_sentiment = {}
topic_count = {}
for idx, topic in topics:
    topic_count[idx] = 0


for index, doc in enumerate(corpus_tfidf):
    _topics = lda_model_tfidf.get_document_topics(doc)
    topic_max = max(_topics, key=lambda x:x[1])

    if topic_max[1] < 0.3: continue

    topic_count[topic_max[0]] += 1

    if topic_max[0] not in topic_docs:
        topic_docs[topic_max[0]] = []
        topic_sentiment[topic_max[0]] = {'pos': 0, 'neg': 0, 'neu': 0, 'compound': 0}

    score = sid.polarity_scores(dataMap[index]['text'])
    topic_sentiment[topic_max[0]]['pos'] += score['pos']
    topic_sentiment[topic_max[0]]['neu'] += score['neu']
    topic_sentiment[topic_max[0]]['neg'] += score['neg']


    topic_docs[topic_max[0]].append(dataMap[index])


sum_topic = 0
for idx, topic in topics:
    sum_topic += topic_count[idx]

for idx, topic in topics:
    print('Topic: {} Word: {}'.format(idx, topic))
    print('Topic percentage: {}'.format(topic_count[idx]/float(sum_topic)))

dataMap_pos = sorted(dataMap, key=lambda x: x['score']['pos'], reverse=True)
print('============================================================')
print('most pos messages:')
for _d in dataMap_pos[0:20]:
    print(_d)

dataMap_neg = sorted(dataMap, key=lambda x: x['score']['neg'], reverse=True)
print('============================================================')
print('most neg messages:')
for _d in dataMap_neg[0:20]:
    print(_d)



for k, v in topic_docs.items():
    print('=============================================')
    print('Topic: {}'.format(k))
    sum = topic_sentiment[k]['neg'] + topic_sentiment[k]['pos'] + topic_sentiment[k]['neu']
    print('sentiment: pos {} neu {} neg {}'.format(topic_sentiment[k]['pos']/float(sum) ,topic_sentiment[k]['neu']/float(sum) ,topic_sentiment[k]['neg']/float(sum)))
    for _d in v:
        print(_d['text'])
        print('\n')
