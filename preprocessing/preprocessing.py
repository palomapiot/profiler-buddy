import os
import pandas as pd
import emoji
import re
import numpy as np

from emoji import UNICODE_EMOJI
from ast import literal_eval
from itertools import chain

import spacy
nlp = spacy.load('en_core_web_lg')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, TweetTokenizer, casual_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from gensim import corpora, models
from gensim.models import Phrases

import pyphen
PYPHEN_DIC = pyphen.Pyphen(lang='en')

from collections import Counter, OrderedDict, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer()
def cosine_similarity_sklearn(documents):
    X_train_counts = tfidf_vectorizer.fit_transform(documents)
    similarities = cosine_similarity(X_train_counts) 
    return similarities.mean()

all_pos_tags = ["NO_TAG", "ADJ", "ADP", "ADV","AUX", "CONJ","CCONJ","DET",
                      "INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM",
                      "VERB","X","EOL","SPACE"]
pos_tags_sorted = sorted(all_pos_tags)

def POS_tags(text):
    text = nlp(text)
    c = Counter()
    c.update({x:0 for x in all_pos_tags})
    pos_list = [token.pos_ for token in text]
    assert len(set(pos_list).difference(set(all_pos_tags))) == 0
    c.update(pos_list)
    c = OrderedDict(sorted(c.items(), key=lambda e: e[0]))
    return list(c.values())

def count_emoji(text):
    return len([c for c in text if c in UNICODE_EMOJI])/len(text)


def face_smiling(text):
    return len([c for c in text if c in '😀😃😄😁😆😅🤣😂🙂🙃😉😊😇'])/len(text)


def face_affection(text):
    return len([c for c in text if c in '🥰😍🤩😘😗☺😚😙'])/len(text)


def face_tongue(text):
    return len([c for c in text if c in '😋😛😜🤪😝🤑'])/len(text)


def face_hand(text):
    return len([c for c in text if c in '🤗🤭🤫🤔'])/len(text)


def face_neutral_skeptical(text):
    return len([c for c in text if c in '🤐🤨😐😑😶😏😒🙄😬🤥'])/len(text)


def face_concerned(text):
    return len([c for c in text if c in '😕😟🙁☹😮😯😲😳🥺😦😧😨😰😥😢😭😱😖😣😞'])/len(text)


def monkey_face(text):
    return len([c for c in text if c in '🙈🙉🙊'])/len(text)


def love(text):
    return len([c for c in text if c in '💋💌💘💝💖💗💓💞💕💟❣💔❤🧡💛💚💙💜🤎🖤'])/len(text)

def sentences_count(tweet):
    sentences = sent_tokenize(tweet)
    return len(sentences)/len(tweet)

def words_count(tweet):
  tknzr = TweetTokenizer()
  tweet_words = tknzr.tokenize(tweet)
  return len(tweet_words), tweet_words

# Using Flesch-Kincaid readability tests (high score -> more easy to read)
def readability(tweet):
    totalSentences = float(sentences_count(tweet))
    totalTweetWords, tweetWords = words_count(tweet)
    totalTweetWords = float(totalTweetWords)
    regexpURL = re.compile(r'https?:\/\/.[^\s]*|www\.[^\s]*')
    totalSyllables = 0.0
    for word in tweetWords:
        if not regexpURL.search(word):
            hyphenated = PYPHEN_DIC.inserted(word)
            syllables = hyphenated.count("-") + 1 - hyphenated.count("--")
            totalSyllables += syllables
    if totalSentences > 0 and totalTweetWords > 0:
        score = 206.835 - 1.015 * (totalTweetWords/totalSentences) - 84.6 * (totalSyllables/totalTweetWords)
    else:
        print("Readability issue")
        score = 0.0
    return score

def repeated_alphabets(text):
    tknzr = TweetTokenizer()
    tweet_words = tknzr.tokenize(text)
    floodings = 0
    regexpURL = re.compile(r'https?:\/\/.[^\s]*|www\.[^\s]*')
    for word in tweet_words:
        if not regexpURL.search(word) and len(re.findall(r'(\w)\1{2,}',word)) > 0: 
            floodings += 1
    return floodings/len(text)

def words_repeated(tweet):
    frequency = defaultdict(int)
    tweet = nlp(tweet)
    for token in tweet:
        if token.lemma_ not in spacy_stopwords:
            frequency[token.lemma_] += 1
    maxAppearance = 0
    if len(frequency.items()) > 0:
        maxAppearance = max(frequency.items(), key=operator.itemgetter(1))[1]
    return maxAppearance/len(text)

def exasperation_count(text):
    return len([c for c in text if c in ['ugh', 'mm', 'mmm', 'mmmm', 'hm', 'hmm', 'hmmm', 'ah', 'ahh', 'ahhhh', 'grrr', 'argh', 'sheesh']])/len(text)

def word_len_mean(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words)

def tfidf(text):
  corpus = nltk.tokenize.sent_tokenize(text)
  vectorizer = TfidfVectorizer(min_df=1)
  model = vectorizer.fit_transform(corpus)
  return model.todense().round(2)

def sentiment_analysis(text):
  sid = SentimentIntensityAnalyzer()
  scores = sid.polarity_scores(text)
  return pd.Series([scores['compound'], scores['neu']])

def i_me_my_count(text):
  return len([c for c in text if c in ['i', 'I', 'me', 'my', 'Me', 'My', 'ME', 'MY']])/len(text)

def article_count(text):
  doc = nlp(text)
  return len([token.lemma_ for token in doc if token.pos_ == "DET"])/len(text)

def articles_pron_count(text):
  doc = nlp(text)
  prons = [token.lemma_ for token in doc if token.lemma_ == "-PRON-"]
  return pd.Series([len([c for c in prons if c in ['he', 'she', 'it', 'they', 
                                        'him', 'her', 'them', 
                                        'his', 'her', 'its', 'their',
                                        'hers', 'theirs',
                                        'himself', 'herself', 'itself', 'themselves', 'themself']])/len(text), 
                    len([token.lemma_ for token in doc if token.pos_ == "DET"])/len(text)])

def word_num_mean(text):
  return sum([len(casual_tokenize(t)) for t in text]) * 1. / len(text)


def preprocess(author_id, comments):
    print('preprocessing data')
    data = pd.DataFrame([[author_id, comments]], columns=['author_id', 'text'], index=[0]) 

    ### Emojis ###
    data['face_smiling'] = data['text'].apply(face_smiling)
    data['face_affection'] = data['text'].apply(face_affection)
    data['face_tongue'] = data['text'].apply(face_tongue)
    data['face_hand'] = data['text'].apply(face_hand)
    data['face_neutral_skeptical'] = data['text'].apply(face_neutral_skeptical)
    data['face_concerned'] = data['text'].apply(face_concerned)
    data['monkey_face'] = data['text'].apply(monkey_face)
    data['love'] = data['text'].apply(love)
    data['emoji_count'] = data['text'].apply(count_emoji)

    ### URLs / web links ###
    data['url_count'] = data.text.apply(lambda x: len(re.findall('http\S+', x))/len(x))

    ### Hashtag ###
    data['hash_count'] = data['text'].apply(lambda x: len(re.findall('[#]', x))/len(x))

    ### Semicolon ###
    data['semicolon_count'] = data['text'].apply(lambda x: len(re.findall('[;]', x))/len(x))

    ### Ellipsis count ###
    data['ellipsis_count'] = data['text'].apply(lambda x: len(re.findall('[...]', x))/len(x))

    ### SA: compound & neutral ###
    data[['compound_sentiment_analysis', 'neutral_sentiment_analysis']] = data['text'].apply(sentiment_analysis)

    ### I, me, my ###
    data['i_me_my'] = data['text'].apply(i_me_my_count)

    ### Word count ### 
    data['word_count'] = data['text'].apply(lambda x: len(re.findall('[a-zA-Z]', x))/len(x))

    ### Articles count ###
    data['article_count'] = data['text'].apply(article_count)

    ### Third person pronouns & Articles ###
    data[['third_person_pron_count', 'article_count']] = data['text'].apply(articles_pron_count)

    ### Puntuation ###
    data['space_count'] = data['text'].apply(lambda x: len(re.findall(' ', x))/len(x))

    data['line_count'] = data['text'].apply(lambda x: len(re.findall('\n', x))/len(x))

    data['capital_count'] = data['text'].apply(lambda x: len(re.findall('[A-Z]', x))/len(x))

    data['digits_count'] = data['text'].apply(lambda x: len(re.findall('[0-9]', x))/len(x))

    data['curly_brackets_count'] = data['text'].apply(lambda x: len(re.findall('[\{\}]', x))/len(x))

    data['round_brackets_count'] = data['text'].apply(lambda x: len(re.findall('[\(\)]', x))/len(x))

    data['square_brackets_count'] = data['text'].apply(lambda x: len(re.findall('\[\]', x))/len(x))

    data['underscore_count'] = data['text'].apply(lambda x: len(re.findall('[_]', x))/len(x))

    data['question_mark_count'] = data['text'].apply(lambda x: len(re.findall('[?]', x))/len(x))

    data['exclamation_mark_count'] = data['text'].apply(lambda x: len(re.findall('[!]', x))/len(x))

    data['dollar_mark_count'] = data['text'].apply(lambda x: len(re.findall('[$]', x))/len(x))

    data['ampersand_mark_count'] = data['text'].apply(lambda x: len(re.findall('[&]', x))/len(x))

    data['tag_count'] = data['text'].apply(lambda x: len(re.findall('[@]', x))/len(x))

    data['slashes_count'] = data['text'].apply(lambda x: len(re.findall('[/,\\\\]', x))/len(x))

    data['operator_count'] = data['text'].apply(lambda x: len(re.findall('[+=\-*%<>^|]', x))/len(x))

    data['punc_count'] = data['text'].apply(lambda x: len(re.findall('[\'\",.:;`]', x))/len(x))

    data['sentences_num'] = data['text'].apply(sentences_count)

    data['repeated_alphabets'] = data['text'].apply(repeated_alphabets)

    data['readability_score'] = data['text'].apply(readability)

    data['word_len_mean'] = data['text'].apply(word_len_mean)

    data['word_num_mean'] = data['text'].apply(word_num_mean)

    data['cosine_similarity'] = cosine_similarity_sklearn(data['text'])

    pos = POS_tags(str(data['text']))
    for i in range(len(pos_tags_sorted)):
        data[pos_tags_sorted[i]] = pos[i]/len(data['text'])

    # lda
    print('lda topics')
    data = add_lda_topics(data)

    return data

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
lemmatizer = WordNetLemmatizer()

def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

def add_lda_topics(data):
    # LDA Model
    lda = models.LdaModel.load('classifiers/lda_model/lda_model')

    # get tokens from text
    corpus = data['text'].to_list()
    data['sentences'] = data.text.map(sent_tokenize)
    data['tokens_sentences'] = data['sentences'].map(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
    data['POS_tokens'] = data['tokens_sentences'].map(lambda tokens_sentences: [pos_tag(tokens) for tokens in tokens_sentences])
    data['tokens_sentences_lemmatized'] = data['POS_tokens'].map(
        lambda list_tokens_POS: [
            [
                lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) 
                if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
            ] 
            for tokens_POS in list_tokens_POS
        ]
    )
    stopwords_custom = ['[', ']', 'RT', '#', '@', ',', '.', '!', 'http', 'https']
    my_stopwords = list(spacy_stopwords) + stopwords_custom
    
    data['tokens'] = data['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
    data['tokens'] = data['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() 
                                                    and token.lower() not in my_stopwords and len(token)>1])

    tokens = data['tokens'].tolist()
    bigram_model = Phrases(tokens)
    trigram_model = Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])

    # create new_corpus
    dictionary_LDA = corpora.Dictionary(tokens)
    unseen_corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]

    # run model on new_corpus
    np.random.seed(123456)
    num_topics = 20
    lda_model = models.LdaModel(unseen_corpus, num_topics=num_topics, \
                                    id2word=dictionary_LDA, \
                                    passes=4, alpha=[0.01]*num_topics, \
                                    eta=[0.01]*len(dictionary_LDA.keys()))

    # get document topic and append to df
    topics = [lda_model[unseen_corpus[i]] for i in range(len(data))]

    # like TF-IDF, create a matrix of topic weighting, with documents as rows and topics as columns
    document_topic = \
    pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics]).reset_index(drop=True).fillna(0)

    data = pd.concat([data, document_topic], axis=1, sort=False)
    data = data.drop(['sentences', 'tokens_sentences', 'POS_tokens', 'tokens_sentences_lemmatized', 'tokens'], 1)
    
    return data