"""
    Copyright 2020-2021 Paloma Piot Pérez-Abadín
	
	This file is part of profiler-buddy.
    profiler-buddy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    profiler-buddy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with profiler-buddy.  If not, see <https://www.gnu.org/licenses/>.
"""

# TODO: delete me
import os
import pandas as pd
import emoji
import re
from emoji import UNICODE_EMOJI
from ast import literal_eval

import spacy
nlp = spacy.load('en_core_web_lg')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

import nltk
from nltk.tokenize import sent_tokenize, TweetTokenizer, casual_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('vader_lexicon')

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
pos_tags = sorted(all_pos_tags)

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
    #print(data)
    print(data.text)
    ### Emojis ###
    print('Emojis')
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
    print('URLs')
    data['url_count'] = data.text.apply(lambda x: len(re.findall('http\S+', x))/len(x))

    ### Hashtag ###
    print('Hashtag')
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
    for i in range(len(pos_tags)):
        data[pos_tags[i]] = pos[i]/len(data['text'])
    return data
    

# TODO: LDA topics