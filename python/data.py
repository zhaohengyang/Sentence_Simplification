import pickle
from nltk import tokenize
from collections import Counter
from itertools import chain
import numpy as np
from headline_generator.config import Config1, Config2, Choose_config
import os
import json
import itertools

# todo add spacy tokenization in training
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


def get_vocab(train_gen, val_gen):
    '''Build vocabulary: take each batch articles and add word counter to the total word counter
     '''
    vocabcount = Counter()
    for article_batch in train_gen:
        # build vocabulary
        # article_batch['title'] is a list of titles
        lst = article_batch['title'] + article_batch['content']
        vocabcount += Counter(w for txt in lst for w in txt.split())
    for article_batch in val_gen:
        # build vocabulary
        lst = article_batch['title'] + article_batch['content']
        vocabcount += Counter(w for txt in lst for w in txt.split())

    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab, vocabcount

def get_glove_embedding_matrix(config):
    glove_name, embedding_dim = config.glove_word_embedding_300["glove_name"], config.glove_word_embedding_300["embedding_dim"]
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

def get_word_embedding_from_valcabulary(glove_index_dict, glove_embedding_weights, idx2word, word2idx, config):
    '''
    :param glove_index_dict: word to index of glove_embedding_weights (40,000, 300)
    :param glove_embedding_weights: word vectors
    :param idx2word: mapping from index to word (voc_size,)
    :param word2idx: mapping from word to index (voc_size,)
    :param config: settings
    :return:
    embedding: word vectors of the vocabulary (For word in the glove, use it. For word not in the glove, use random.uniform)
    glove_idx2idx: word that not in the glove, map to similar word that in the glove
    '''
    embedding_dim = config.glove_word_embedding_300["embedding_dim"]
    glove_thr = config.glove_word_embedding_300["glove_thr"]
    vocab_size = config.vocab_size
    nb_unknown_words = config.nb_unknown_words

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

    # lots of word in the full vocabulary (word2idx) are outside vocab_size.
    # Build an alterantive which will map them to their closest match in glove
    # but only if the match is good enough (cos distance above glove_thr)
    word2glove = {}
    for w in word2idx:
        if w in glove_index_dict:
            g = w
        elif w.lower() in glove_index_dict:
            g = w.lower()
        elif w.startswith('#') and w[1:] in glove_index_dict:
            g = w[1:]
        elif w.startswith('#') and w[1:].lower() in glove_index_dict:
            g = w[1:].lower()
        else:
            continue
        word2glove[w] = g

    normed_embedding = embedding / np.array([np.sqrt(np.dot(gweight, gweight)) for gweight in embedding])[:, None]
    glove_match = []
    for w, idx in word2idx.iteritems():
        if idx >= vocab_size - nb_unknown_words and w.isalpha() and w in word2glove:
            gidx = glove_index_dict[word2glove[w]]
            gweight = glove_embedding_weights[gidx, :].copy()
            # find row in embedding that has the highest cos score with gweight
            gweight /= np.sqrt(np.dot(gweight, gweight))

            # mapping to vocab: normed_embedding[:vocab_size - nb_unknown_words]
            score = np.dot(normed_embedding[:vocab_size - nb_unknown_words], gweight)
            while True:
                embedding_idx = score.argmax()
                s = score[embedding_idx]
                if s < glove_thr:
                    break
                if idx2word[embedding_idx] in word2glove:
                    glove_match.append((w, embedding_idx, s))
                    break
                score[embedding_idx] = -1
    glove_match.sort(key=lambda x: -x[2])
    print '# of glove substitutes found', len(glove_match)

    # manually check that the worst substitutions we are going to do are good enough
    for orig, sub, score in glove_match[-10:]:
        print score, orig, '=>', idx2word[sub]

    glove_idx2idx = dict((word2idx[w], embedding_idx) for w, embedding_idx, _ in glove_match)
    return embedding, glove_idx2idx

def view_data(n_lines=3, path='../sample_data/train.jsonl'):
    with open(path) as f:
        for next_n_lines in itertools.izip_longest(*[f] * n_lines):
            articles = {"title": [], "content": []}
            for line in next_n_lines:
                if line:
                    json_data = json.loads(line.strip())
                    print "title", json_data['title']
                    print "content", json_data['content']
            break
# todo: check if all the headline exist in raw data
# todo: check out if some headline did summrize the content well
# todo: test article may contain unknown words
def main():
    # Load settings
    config = Choose_config.current_config['class']
    early_stop = config.early_stop
    eos =  config.eos # end of sentence
    empty = config.empty # RNN mask of no data
    vocab_size = config.vocab_size
    nb_unknown_words = config.nb_unknown_words
    FN0 = config.FN0
    oov0 = config.oov0
    nb_val_samples = config.nb_val_samples
    batch_size = config.batch_size
    unkonwn_word = config.unkonwn_word
    start_word_index = config.start_word_index
    train_path = '../sample_data/train.jsonl'
    val_path = '../sample_data/test.jsonl'


    # Load articles
    print("load articles")
    train_gen = load_article(raw_path = train_path, early_stop=early_stop, batch_size=batch_size)
    val_gen = load_article(raw_path = val_path, early_stop=nb_val_samples, batch_size=batch_size)
    # articles = train_gen.next()
    # for i,j in zip(articles['title'], articles['content']):
    #     print("head:",i)
    #     print('content:', j)
    # Tokenize articles
    # title:first sentence in article
    # tok_articles = {'title': articles['title'], 'content': [tokenize.sent_tokenize(c)[0] for c in articles['content']]}
    # title: all sentence in article

    # build vocabulary
    print("build vocabulary")
    vocab, vocabcount = get_vocab(train_gen, val_gen)
    # print(articles['content'][0])
    # print(" ".join(tokenize.sent_tokenize(articles['content'][0])))
    print("vocab_size in setting:", vocab_size)
    print("word found in sample:", len(vocab))
    assert len(vocab) >= vocab_size

    # def add_word_to_vocab(word):
    #     index = max(idx2word.keys(), key=int) + 1
    #     idx2word[index] = word
    #     word2idx[word] = index

    # Word indexing
    print("Create word indexing")
    word2idx = dict((word, idx + start_word_index) for idx, word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    word2idx['<unkonwn_word>'] = unkonwn_word

    idx2word = dict((idx, word) for word, idx in word2idx.iteritems())
    idx2word[empty] = '_'
    idx2word[eos] = '~'
    idx2word[unkonwn_word] = 'unkonwn_word'

    # last nb_unknown words can be used as place holders for unknown/oov words
    for i in range(nb_unknown_words):
        idx2word[vocab_size - 1 - i] = '<%d>' % i

    for i in range(oov0, len(idx2word)):
        idx2word[i] = idx2word[i] + '^'

    # # Save word index into a pickle file
    # X = [[word2idx[token] for token in d.split()] for d in tok_articles['content']]
    # Y = [[word2idx[token] for token in headline.split()] for headline in tok_articles['title']]
    # with open('../sample_data/%s.data.pkl' % FN0, 'wb') as fp:
    #     pickle.dump((X, Y), fp, -1)

    print("Generate word embedding")
    # load glove matrix from glove source file
    glove_index_dict, glove_embedding_weights = get_glove_embedding_matrix(config)
    # Use glove matrix to create a word embedding matrix of our vocabulary
    embedding, glove_idx2idx = get_word_embedding_from_valcabulary(glove_index_dict, glove_embedding_weights, idx2word, word2idx, config)

    # Save word embedding into a pickle file
    with open('../sample_data/%s.pkl' % FN0, 'wb') as fp:
        pickle.dump((embedding, idx2word, word2idx, glove_idx2idx), fp, -1)

if __name__ == "__main__":
    main()
    # view_data()


