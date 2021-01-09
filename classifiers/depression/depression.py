import numpy as np
import os 
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

from official.nlp import optimization
from official.nlp.bert.tokenization import FullTokenizer
from pyemd import emd
import gensim.downloader as api

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

NB_BATCHES_TRAIN = 2000
INIT_LR = 5e-5 # initial learning rate
WARMUP_STEPS = int(NB_BATCHES_TRAIN * 0.1) 

def squad_loss_fn(labels, model_outputs):
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs

    start_loss = tf.keras.backend.sparse_categorical_crossentropy(
        start_positions, start_logits, from_logits=True)
    end_loss = tf.keras.backend.sparse_categorical_crossentropy(
        end_positions, end_logits, from_logits=True)
    
    total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2

    return total_loss

optimizer = optimization.create_optimizer(
    init_lr=INIT_LR,
    num_train_steps=NB_BATCHES_TRAIN,
    num_warmup_steps=WARMUP_STEPS)

bert_squad = load_model(DIR_PATH + '/bert_squad_model', compile=False)
bert_squad.compile(optimizer, squad_loss_fn)

### Prediction Utils ###
my_bert_layer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    trainable=False)
vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

def is_whitespace(c):
    '''
    Tell if a chain of characters corresponds to a whitespace or not.
    '''
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def whitespace_split(text):
    '''
    Take a text and return a list of "words" by splitting it according to
    whitespaces.
    '''
    doc_tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens

def tokenize_context(text_words):
    '''
    Take a list of words (returned by whitespace_split()) and tokenize each word
    one by one. Also keep track, for each new token, of its original word in the
    text_words parameter.
    '''
    text_tok = []
    tok_to_word_id = []
    for word_id, word in enumerate(text_words):
        word_tok = tokenizer.tokenize(word)
        text_tok += word_tok
        tok_to_word_id += [word_id]*len(word_tok)
    return text_tok, tok_to_word_id

def get_ids(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)

def get_mask(tokens):
    return np.char.not_equal(tokens, "[PAD]").astype(int)

def get_segments(tokens):
    seg_ids = []
    current_seg_id = 0
    for tok in tokens:
        seg_ids.append(current_seg_id)
        if tok == "[SEP]":
            current_seg_id = 1-current_seg_id # turns 1 into 0 and vice versa
    return seg_ids

def create_input_dict(question, context):
    '''
    Take a question and a context as strings and return a dictionary with the 3
    elements needed for the model. Also return the context_words, the
    context_tok to context_word ids correspondance and the length of
    question_tok that we will need later.
    '''
    question_tok = tokenizer.tokenize(question)

    context_words = whitespace_split(context)
    context_tok, context_tok_to_word_id = tokenize_context(context_words)

    input_tok = question_tok + ["[SEP]"] + context_tok + ["[SEP]"]
    input_tok += ["[PAD]"]*(384-len(input_tok)) # in our case the model has been
                                                # trained to have inputs of length max 384
    input_dict = {}
    input_dict["input_word_ids"] = tf.expand_dims(tf.cast(get_ids(input_tok), tf.int32), 0)
    input_dict["input_mask"] = tf.expand_dims(tf.cast(get_mask(input_tok), tf.int32), 0)
    input_dict["input_type_ids"] = tf.expand_dims(tf.cast(get_segments(input_tok), tf.int32), 0)

    return input_dict, context_words, context_tok_to_word_id, len(question_tok)

### Similarity between the answers and the predicted answer ###

word_vectors = api.load("glove-wiki-gigaword-100")
def find_best_answer(predicted_answer, answers, is_special):
    similarities = []
    for answer in answers:
        similarities.append(word_vectors.wmdistance(predicted_answer, answer))
    # lowest wmd, more similar
    idx = similarities.index(min(similarities))
    # return idx (question value)
    if is_special == True:
        if idx == 1 or idx == 2:
            idx = 1
        elif idx == 3 or idx == 4:
            idx = 2
        elif idx == 5 or idx == 6:
            idx = 3
    return idx

### Questions ###

questions = {
    1: '''Do you feel sad?''', # Sadness
    2: '''What do you think about your future?''', # Pessimism
    3: '''Past Failure''',
    4: '''Do you enjoy things?''', # Loss of pleasure
    5: '''Guilty Feelings''', # Guilty Feelings
    6: '''Punishment''', # Punishment Feelings
    7: '''What do you think about yourself?''', # Self-Dislike -> both works
    8: '''Do you blame yourself?''', # Self-Criticalness
    9: '''Do you think of killing yourself?''', # Suicidal Thoughts or Wishes
    10: '''Do you cry?''', # Crying
    11: '''Are you restless?''', # Agitation
    12: '''Have you lost interest?''', # Loss of Interest
    13: '''Are you indecisive?''', # Indecisiveness
    14: '''Are you wothless?''', # Worthlessness --------
    15: '''Do you have energy?''', # Loss of Energy
    16: '''How is your sleep?''', # Changes in Sleeping Pattern
    17: '''Are you irritable?''', # Irritability
    18: '''How is your appetite?''', # Changes in Appetite
    19: '''Can you concentrate?''', # Concentration Difficulty
    20: '''Are you tired?''', # Tiredness or Fatigue --------
    21: '''Are you interested in sex?''' # Loss of Interest in Sex
}

questions_answers = {
    1: ["I do not feel sad", "I feel sad much of the time", "I am sad all the time", "I am so sad or unhappy that I can't stand it"], # Sadness
    2: ["I am not discouraged about my future", "I feel more discouraged about my future than I used to be", "I do not expect things to work out for me", "I feel my future is hopeless and will only get worse"], # Pessimism
    3: ["I do not feel like a failure", "I have failed more than I should have", "As I look back, I see a lot of failures", "I feel I am a total failure as a person"],
    4: [" get as much pleasure as I ever did from the things I enjoy", "I don't enjoy things as much as I used to", "I get very little pleasure from the things I used to enjoy", "I can't get any pleasure from the things I used to enjoy"], # Loss of pleasure
    5: ["I don't feel particularly guilty", "I feel guilty over many things I have done or should have done", "I feel quite guilty most of the time", "I feel guilty all of the time"], # Guilty Feelings
    6: ["I don't feel I am being punished", "I feel I may be punished", "I expect to be punished", "I feel I am being punished"], # Punishment Feelings
    7: ["I feel the same about myself as ever", "I have lost confidence in myself", "I am disappointed in myself", "I dislike myself"], # Self-Dislike -> both works
    8: ["I don't criticize or blame myself more than usual", "I am more critical of myself than I used to be", "I criticize myself for all of my faults", "I blame myself for everything bad that happens"], # Self-Criticalness
    9: ["I don't have any thoughts of killing myself", "I have thoughts of killing myself, but I would not carry them out", "I would like to kill myself", "I would kill myself if I had the chance"], # Suicidal Thoughts or Wishes
    10: ["I don't cry anymore than I used to", "I cry more than I used to", "I cry over every little thing", "I feel like crying, but I can't"], # Crying
    11: ["I am no more restless or wound up than usual", "I feel more restless or wound up than usual", "I am so restless or agitated that it's hard to stay still", "I am so restless or agitated that I have to keep moving or doing something"], # Agitation
    12: ["I have not lost interest in other people or activities", "I am less interested in other people or things than before", "I have lost most of my interest in other people or things", "It's hard to get interested in anything"], # Loss of Interest
    13: ["I make decisions about as well as ever", "I find it more difficult to make decisions than usual", "I have much greater difficulty in making decisions than I used to", "I have trouble making any decisions"], # Indecisiveness
    14: ["I do not feel I am worthless", "I don't consider myself as worthwhile and useful as I used to", "I feel more worthless as compared to other people", "I feel utterly worthless"], # Worthlessness
    15: ["I have as much energy as ever", "I have less energy than I used to have", "I don't have enough energy to do very much", "I don't have enough energy to do anything"], # Loss of Energy
    16: ["I have not experienced any change in my sleeping pattern", "I sleep somewhat more than usual", "I sleep somewhat less than usual", "I sleep a lot more than usual", "I sleep a lot less than usual", "I sleep most of the day", "I wake up 1-2 hours early and can't get back to sleep"], # Changes in Sleeping Pattern
    17: ["I am no more irritable than usual", "I am more irritable than usual", "I am much more irritable than usual", "I am irritable all the time"], # Irritability
    18: ["I have not experienced any change in my appetite", "My appetite is somewhat less than usual", "My appetite is somewhat greater than usual", "My appetite is much less than before", "My appetite is much greater than usual", "I have no appetite", "I crave food all the time"], # Changes in Appetite
    19: ["I can concentrate as well as ever", "I can't concentrate as well as usual", "It's hard to keep my mind on anything for very long", "I find I can't concentrate on anything"], # Concentration Difficulty
    20: ["I am no more tired or fatigued than usual", "I get more tired or fatigued more easily than usual", "I am too tired or fatigued to do a lot of the things I used to do", "I am too tired or fatigued to do most of the things I used to do"], # Tiredness or Fatigue
    21: ["I have not noticed any recent change in my interest in sex", "I am less interested in sex than I used to be", "I am much less interested in sex now", "I have lost interest in sex completely at all"] # Loss of Interest in Sex
}

def find_answers(comments):
    answers = {
        "questionnaire": {},
        "questionnaire_reasons": {}
    }
    for idx in range(1, 22):
        my_question = questions[idx]
        for my_context in comments:
            # creation
            my_input_dict, my_context_words, context_tok_to_word_id, question_tok_len = create_input_dict(my_question, my_context)
            # prediction
            start_logits, end_logits = bert_squad(my_input_dict, training=False)
            # remove the ids corresponding to the question and the ["SEP"] token
            start_logits_context = start_logits.numpy()[0, question_tok_len+1:]
            end_logits_context = end_logits.numpy()[0, question_tok_len+1:]
            # interpretation
            pair_scores = np.ones((len(start_logits_context), len(end_logits_context)))*(-1E10)
            for i in range(len(start_logits_context-1)):
                for j in range(i, len(end_logits_context)):
                    pair_scores[i, j] = start_logits_context[i] + end_logits_context[j]
            pair_scores_argmax = np.argmax(pair_scores)
            # ensure that the conterxt is inside our context
            if len(context_tok_to_word_id) > pair_scores_argmax // len(start_logits_context) and len(context_tok_to_word_id) > pair_scores_argmax % len(end_logits_context):
                start_word_id = context_tok_to_word_id[pair_scores_argmax // len(start_logits_context)]
                end_word_id = context_tok_to_word_id[pair_scores_argmax % len(end_logits_context)]
            else:
                start_word_id = 0
                end_word_id = 0
            # answer context
            predicted_answer = ' '.join(my_context_words[start_word_id:end_word_id+1])
            marked_text = str(my_context.replace(predicted_answer, f"<strong>{predicted_answer}</strong>"))
            # set the higher pair_scores_argmax to question
            questionnaire = answers["questionnaire"]
            q_idx = 'q' + str(idx)
            if q_idx not in questionnaire.keys() or pair_scores_argmax > int(questionnaire[q_idx]["score"]):
                questionnaire[q_idx] = {"context": marked_text, "score": str(pair_scores_argmax)}
        # get most similar answer
        is_special = False
        if idx == 16 or idx == 18:
            is_special = True
        questionnaire_answer = find_best_answer(predicted_answer, questions_answers[idx], is_special)
        answers["questionnaire"][q_idx] = questionnaire_answer
    answers["questionnaire_reasons"] = questionnaire
    return answers