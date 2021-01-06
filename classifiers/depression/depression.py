import tensorflow as tf
import tensorflow_hub as hub

from official.nlp.bert.tokenization import FullTokenizer
import numpy as np

### Model ###
class BertSquadLayer(tf.keras.layers.Layer):

  def __init__(self):
    super(BertSquadLayer, self).__init__()
    self.final_dense = tf.keras.layers.Dense(
        units=2,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))

  def call(self, inputs):
    logits = self.final_dense(inputs) # (batch_size, seq_len, 2)

    logits = tf.transpose(logits, [2, 0, 1]) # (2, batch_size, seq_len)
    unstacked_logits = tf.unstack(logits, axis=0) # [(batch_size, seq_len), (batch_size, seq_len)] 
    return unstacked_logits[0], unstacked_logits[1]

class BERTSquad(tf.keras.Model):
    
    def __init__(self,
                 name="bert_squad"):
        super(BERTSquad, self).__init__(name=name)
        
        # in a prod env, pick the large bert
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            trainable=True)
        
        self.squad_layer = BertSquadLayer()
    
    def apply_bert(self, inputs):
        _ , sequence_output = self.bert_layer([inputs["input_word_ids"],
                                               inputs["input_mask"],
                                               inputs["input_type_ids"]])
        return sequence_output

    def call(self, inputs):
        seq_output = self.apply_bert(inputs)

        start_logits, end_logits = self.squad_layer(seq_output)
        
        return start_logits, end_logits

bert_squad = BERTSquad()

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

def find_answers(my_context):
    answers = {}
    for idx in range(1, 22):
        # creation
        my_question = questions[idx]
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
        start_word_id = context_tok_to_word_id[pair_scores_argmax // len(start_logits_context)]
        end_word_id = context_tok_to_word_id[pair_scores_argmax % len(end_logits_context)]
        # answer
        predicted_answer = ' '.join(my_context_words[start_word_id:end_word_id+1])
        answers[idx] = predicted_answer
    return answers