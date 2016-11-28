import itertools
import json
from collections import Counter
from nltk import tokenize
from tensorflow.python.platform import gfile
import re
import os
import numpy as np
from nltk import tokenize as nltk_tokenizor
from spacy.en import English
parser = English()

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
_OOV = b"_OOV"
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")




def load_article(raw_path, early_stop, batch_size):
    '''
    :param raw_path:
    :param early_stop:
    :param batch_size:
    :return: yeild tokenized batch of articles
    '''
    article_index = 0
    sample_file = open(raw_path)

    if early_stop:
        with open(raw_path) as f:
            for next_n_lines in itertools.izip_longest(*[f] * batch_size):
                articles = {"title": [], "content": []}
                for line in next_n_lines:
                    if line:
                        json_data = json.loads(line.strip())
                        articles["title"].append(json_data['title'])
                        articles["content"].append(json_data['content'])
                    early_stop -= 1
                    if early_stop <= 0:
                        break

                tok_articles = {'title': [" ".join(tokenize.sent_tokenize(c)) for c in articles['title']],
                                    'content': [" ".join(tokenize.sent_tokenize(c)) for c in articles['content']]}
                # print("len(tok_articles['content']", len(tok_articles['content']))
                yield tok_articles

                if early_stop <= 0:
                    break
    else:
        with open(raw_path) as f:
            for next_n_lines in itertools.izip_longest(*[f] * batch_size):
                articles = {"title": [], "content": []}
                for line in next_n_lines:
                    if line:
                        json_data = json.loads(line.strip())
                        articles["title"].append(json_data['title'])
                        articles["content"].append(json_data['content'])

                tok_articles = {'title': [" ".join(tokenize.sent_tokenize(c)) for c in articles['title']],
                                'content': [" ".join(tokenize.sent_tokenize(c)) for c in articles['content']]}
                yield tok_articles




import unicodedata
# def create_text_data(raw_path, content_path, title_path):
#     with gfile.GFile(raw_path, mode="rb") as raw:
#         with gfile.GFile(content_path, mode="wb") as content_file:
#             with gfile.GFile(title_path, mode="wb") as title_file:
#
#                 for i, line in enumerate(raw):
#                     if i > 0:
#                         break
#
#                     if line:
#                         json_data = json.loads(line.strip())
#                         # print("create_text_data: "+json_data['title'])
#
#                         decoded = unicodedata.normalize('NFKD', json_data['content']).encode('ASCII', 'ignore')
#                         content_list = decoded.split("\n")
#                         print(json_data['content'].split("\n"))
#                         print(content_list)
#                         content_line = " ".join(nltk_tokenizor.sent_tokenize("".join(content_list)))
#                         decoded = unicodedata.normalize('NFKD', json_data['title']).encode('ASCII', 'ignore')
#                         title_line = "".join(decoded.split("\n"))
#                         title_file.write(title_line + b"\n")
#                         content_file.write(content_line + b"\n")

def create_text_data(raw_path, content_path, title_path, sentence_truncate=None, tokenizer=None, normalize_digits=False):
    print("start create_text_data...")
    counter = 0
    with gfile.GFile(raw_path, mode="rb") as raw:
        with gfile.GFile(content_path, mode="wb") as content_file:
            with gfile.GFile(title_path, mode="wb") as title_file:

                for i, line in enumerate(raw):
                    # if i >= 100:
                    #     break

                    counter += 1
                    if counter % 100000 == 0:
                        print("Create vocab: processing line %d" % counter)
                    json_data = json.loads(line.strip())
                    for sentence, file in zip((json_data['title'], json_data['content']),(title_file, content_file)):
                        if sentence_truncate:
                            # sentence = "".join("".join(sentence.split("\n")).split("\r"))
                            sentence = ''.join(sentence.splitlines())
                            sentence = truncate_article(sentence, sentence_truncate)
                        else:
                            sentence = ''.join(sentence.splitlines())
                            # sentence = "".join("".join(sentence.split("\n")).split("\r"))
                        tokens = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)


                        if normalize_digits:
                            tokens = [re.sub(_DIGIT_RE, b"0", w) for w in tokens]
                        output = " ".join(tokens)+ b"\n"
                        file.write(output.encode("utf8"))


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w.lower() for w in words if w]


def spacy_tokenizer(sentence):
    parsedData = parser(sentence)
    return [word.orth_.lower() for word in parsedData]

def truncate_article(article_content, sentence_num):
    if not isinstance(article_content, unicode):
        article_content = article_content.decode("utf8")
    parsedData = parser(article_content)
    sents = []
    for span in parsedData.sents:
        # go from the start to the end of each span, returning each token in the sentence
        # combine each token using join()
        sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
        sents.append(sent)
    return " ".join(sents[:sentence_num])

def get_vocab_counter(json_file_path, sentence_truncate=None, tokenizer=None, normalize_digits=True):
    counter = 0
    vocab = Counter()
    with gfile.GFile(json_file_path, mode="rb") as f:
        for line in f:
            counter += 1
            if counter % 100000 == 0:
                print("Create vocab: processing line %d" % counter)
            json_data = json.loads(line.strip())
            for sentence in (json_data['title'], json_data['content']):
                if sentence_truncate:
                    sentence = truncate_article(sentence, sentence_truncate)
                tokens = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)
                if normalize_digits:
                    tokens = [re.sub(_DIGIT_RE, b"0", w) for w in tokens]

                for token in tokens:
                    vocab[token] += 1
    return vocab

# modified
def create_vocabulary(vocabulary_path, train_json_path, evl_json_path,
                      max_vocabulary_size, oov_size, sentence_truncate=None, tokenizer=None, normalize_digits=True):

    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      oov_size: oov_size out of vocabulary, it is used for keeping unknown words in the inputs
      sentence_truncate: number of sentence selected from the begining of the article
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s" % (vocabulary_path))
        vocabcount = Counter()
        vocabcount += get_vocab_counter(train_json_path, sentence_truncate=sentence_truncate, tokenizer=None, normalize_digits=True)
        vocabcount += get_vocab_counter(evl_json_path, sentence_truncate=sentence_truncate, tokenizer=None, normalize_digits=True)

        vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
        vocab_list = _START_VOCAB + vocab

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size-oov_size]
        for i in xrange(oov_size):
            oov_word = _OOV+"{}".format(i)
            vocab_list.append(oov_word)
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w.encode('utf8') + b"\n")


def sentence_to_token_ids(sentence, vocabulary, oov_size, tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      oov_size: oov_size out of vocabulary, it is used for keeping unknown words in the inputs
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    print(words)
    # todo assigned unknown words to placeholders
    if oov_size > 0:
        sentence_ids = []
        unknown_to_placeholders = {}
        current_oov_index = 0
        for w in words:
            if normalize_digits:
                w = re.sub(_DIGIT_RE, b"0", w)
            if w not in vocabulary.keys():
                # For the same unkonwn words, use the same placeholders
                if w in unknown_to_placeholders.keys():
                    sentence_ids.append(unknown_to_placeholders[w])
                # For a new unknown word, assign a new placeholder
                elif current_oov_index < oov_size:
                    newplaceholder = _OOV+"{}".format(current_oov_index)
                    id = vocabulary.get(newplaceholder, UNK_ID)
                    unknown_to_placeholders[newplaceholder] = id
                    sentence_ids.append(id)
                    current_oov_index += 1
                # For a new unknown word and no more free placeholder, assign the word id to UNK_ID
                else:
                    sentence_ids.append(UNK_ID)
            else:
                sentence_ids.append(vocabulary[w])
        return sentence_ids
    else:
        if not normalize_digits:
            return [vocabulary.get(w, UNK_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)



def data_to_token_ids(json_data_path, title_path, content_path, vocabulary_path, oov_size, sentence_truncate=None,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(title_path):
        print("Tokenizing data in %s" % json_data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(json_data_path, mode="rb") as data_file:
            with gfile.GFile(title_path, mode="wb") as title_file:
                with gfile.GFile(content_path, mode="wb") as content_file:
                    counter = 0
                    for line in data_file:
                        counter += 1
                        if counter % 100000 == 0:
                            print("  tokenizing line %d" % counter)
                        json_data = json.loads(line.strip())
                        for sentence, file in zip([json_data['title'].encode("utf8"), json_data['content'].encode("utf8")],[title_file, content_file]) :
                            if sentence_truncate:
                                sentence = truncate_article(sentence, sentence_truncate)

                            token_ids = sentence_to_token_ids(sentence, vocab, oov_size, tokenizer,
                                                              normalize_digits)

                            output = " ".join([str(tok) for tok in token_ids]) + "\n"
                            file.write(output.encode('utf8'))
    else:
        print("Use exist file:", title_path)
        print("Use exist file:", content_path)



# modified
def prepare_headline_generation_data(data_dir, vocabulary_size, oov_size, sentence_truncate=None, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    vocabulary_size: size of the vocabulary to create and use.
    oov_size: oov_size out of vocabulary, it is used for keeping unknown words in the inputs
    sentence_truncate: number of sentence selected from the begining of the article
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """


  train_content_path = os.path.join(data_dir, "train_content")
  train_title_path = os.path.join(data_dir, "train_title")
  evl_content_path = os.path.join(data_dir, "evl_content")
  evl_title_path = os.path.join(data_dir, "evl_title")

  train_json_path = '../../sample_data/train.jsonl'
  evl_json_path = '../../sample_data/test.jsonl'

  if gfile.Exists(train_json_path):
      # Create vocabularies of the appropriate sizes.
      vocab_path = os.path.join(data_dir, "vocab_%d" % vocabulary_size)
      create_vocabulary(vocab_path, train_json_path, evl_json_path,
                        vocabulary_size, oov_size, sentence_truncate, tokenizer)

      # Create token ids for the training data.
      train_content_ids_path = train_content_path + (".ids_%d" % vocabulary_size)
      train_title_ids_path = train_title_path + (".ids_%d" % vocabulary_size)
      data_to_token_ids(train_json_path, train_title_ids_path, train_content_ids_path, vocab_path, oov_size, sentence_truncate, tokenizer)

      # Create token ids for the development data.
      evl_content_ids_path = evl_content_path + (".ids_%d" % vocabulary_size)
      evl_title_ids_path = evl_title_path + (".ids_%d" % vocabulary_size)
      data_to_token_ids(evl_json_path, evl_title_ids_path, evl_content_ids_path, vocab_path, oov_size, sentence_truncate, tokenizer)

      return (train_content_ids_path, train_title_ids_path,
              evl_content_ids_path, evl_title_ids_path,
              vocab_path)
  else:
    raise ValueError("Vocabulary file %s not found.", train_content_path)



def get_glove_embedding_matrix(glove_name, embedding_dim):
    with open(glove_name) as f:
        glove_n_symbols = sum(1 for _ in f)

    # Get glove word to vector
    glove_index_dict = {}
    glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
    globale_scale = .1
    with open(glove_name, 'r') as fp:
        i = 0
        for l in fp:
            l = l.strip().split()
            w = l[0] # word
            glove_index_dict[w] = i # word to index of glove_embedding_weights
            glove_embedding_weights[i, :] = map(float, l[1:]) # Save the vector to indexed raw
            i += 1
    # glove word vectors
    glove_embedding_weights *= globale_scale

    return glove_index_dict, glove_embedding_weights

def get_word_embedding_from_valcabulary(glove_index_dict, glove_embedding_weights, idx2word, embedding_dim, vocab_size):
    '''
    :param glove_index_dict: word to index of glove_embedding_weights (40,000, 300)
    :param glove_embedding_weights: word vectors
    :param idx2word: mapping from index to word (voc_size,)
    :param vocab_size: vocabulary size
    :return:
    embedding: word vectors of the vocabulary (For word in the glove, use it. For word not in the glove, use random.uniform)
    glove_idx2idx: word that not in the glove, map to similar word that in the glove
    '''


    # generate random embedding with same scale as glove
    # give vector a simulated values in case couldn't find record in glove
    seed = 42
    np.random.seed(seed)
    shape = (vocab_size, embedding_dim)
    print("debug: in get_word_embedding_from_valcabulary(): word embedding shape", shape)
    scale = glove_embedding_weights.std() * np.sqrt(12) / 2  # uniform and not normal
    embedding = np.random.uniform(low=-scale, high=scale, size=shape)

    # copy from glove weights of words that appear in our short vocabulary (idx2word)
    c = 0
    for i in range(vocab_size):
        # index to word
        w = idx2word[i]
        # word to vector
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
        if g is None and w.startswith('#'):  # glove has no hastags (I think...)
            w = w[1:]
            g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
        if g is not None:
            # save vector in embedding
            embedding[i, :] = glove_embedding_weights[g, :]
            c += 1
    return embedding

import pickle
def get_glove_embedding(vocab_path, embedding_path, glove_dir, size, source_vocab_size):
    embedding = None
    if gfile.Exists(embedding_path):
        # load word embeddings
        with open(embedding_path, 'rb') as fp:
            embedding = pickle.load(fp)
    else:
        _, rev_vocab = initialize_vocabulary(vocab_path)
        glove_index_dict, glove_embedding_weights = get_glove_embedding_matrix(glove_dir, size)
        # Use glove matrix to create a word embedding matrix of our vocabulary
        embedding = get_word_embedding_from_valcabulary(glove_index_dict, glove_embedding_weights,
                                                        rev_vocab, size, source_vocab_size)
        # Save word embedding into a pickle file
        with open(embedding_path, 'wb') as fp:
            pickle.dump((embedding), fp, -1)
    return embedding

def main():
    print("Preprocess initialiated...")
    data_dir = "../sample_data"
    raw_train_path = '../../sample_data/train.jsonl'
    raw_val_path = '../../sample_data/test.jsonl'
    train_content_path = os.path.join(data_dir, "train_content")
    train_title_path = os.path.join(data_dir, "train_title")
    evl_content_path = os.path.join(data_dir, "evl_content")
    evl_title_path = os.path.join(data_dir, "evl_title")

    sentence_trunctate = 5
    create_text_data(raw_train_path, train_content_path, train_title_path, sentence_trunctate, spacy_tokenizer)
    create_text_data(raw_val_path, evl_content_path, evl_title_path, sentence_trunctate, spacy_tokenizer)
    # with gfile.GFile(train_content_path, mode="rb") as f:
    #     for line in f:
    #         print(line.decode("utf8"))



if __name__ == "__main__":
    main()





