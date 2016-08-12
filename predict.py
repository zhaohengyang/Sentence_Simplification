from headline_generator.config import Config1, Config2, Choose_config
from headline_generator.attention_layer import WSimpleContextLayer
from headline_generator.train import get_weight_file
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K
import h5py
from headline_generator.attention_layer import SimpleContextLayer
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
import sys
import Levenshtein
from IPython.core.display import display, HTML
from headline_generator.data import load_article
import time
import pandas as pd
import kenlm


def str_shape(x):
    return 'x'.join(map(str, x.shape))

def inspect_model(model):
    print model.name
    for i, l in enumerate(model.layers):
        print i, 'cls=%s name=%s' % (type(l).__name__, l.name)
        weights = l.get_weights()
        for weight in weights:
            print str_shape(weight),
        print

def load_weights(model, filepath):
    """Modified version of keras load_weights that loads as much as it can
    if there is a mismatch between file and model. It returns the weights
    of the first layer in which the mismatch has happened
    """
    print 'Loading', filepath, 'to', model.name
    flattened_layers = model.layers
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            print name
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            print weight_names
            # simplecontextlayer_1 is lambda layer which is like a lambda function, thus don't have weight
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                try:
                    layer = model.get_layer(name=name)
                except:
                    layer = None
                if not layer:
                    print 'failed to find layer', name, 'in model'
                    print 'weights', ' '.join(str_shape(w) for w in weight_values)
                    print 'stopping to load all other layers'
                    weight_values = [np.array(w) for w in weight_values]
                    break

                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                weight_value_tuples += zip(symbolic_weights, weight_values)
                weight_values = None
        K.batch_set_value(weight_value_tuples)
    return weight_values




class Predict_model(object):
    def __init__(self, config, weight_file_path, word_embedding_file):
        self.config = config
        with open(word_embedding_file, 'rb') as fp:
            self.embedding, self.idx2word, self.word2idx, self.glove_idx2idx = pickle.load(fp)
        print("len(self.idx2word)",len(self.idx2word))
        self.weight_file_path = weight_file_path
        self.model, self.wmodel, self.weights = self.load_model()

    # out very own softmax
    # softmax layer doesn't need to be trained
    def output2probs(self, output):
        # input: (*, 944)
        # weights[0]: (944, vocab_size), weights[1]: (vocab_size,)
        # output: (*, vocab_size)
        output = np.dot(output, self.weights[0]) + self.weights[1]
        output -= output.max()
        output = np.exp(output)
        output /= output.sum()
        return output

    def load_model(self):
        weight_decay = self.config.weight_decay
        regularizer = l2(weight_decay) if weight_decay else None
        vocab_size, embedding_size = self.embedding.shape
        p_emb = self.config.p_emb
        rnn_layers = self.config.rnn_layers
        rnn_size = self.config.rnn_size
        p_W = self.config.p_W
        p_U = self.config.p_U
        p_dense = self.config.p_dense
        activation_rnn_size = self.config.activation_rnn_size
        optimizer = self.config.optimizer


        rnn_model = Sequential()
        rnn_model.add(Embedding(vocab_size, embedding_size,
                                input_length=self.config.maxlen,
                                W_regularizer=regularizer, dropout=p_emb, weights=[self.embedding], mask_zero=True))
        for i in range(rnn_layers):
            lstm = LSTM(rnn_size, return_sequences=True,  # batch_norm=batch_norm,
                        W_regularizer=regularizer, U_regularizer=regularizer,
                        b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
                        )
            rnn_model.add(lstm)
            rnn_model.add(Dropout(p_dense))

        # attention layter doesn't need to be trained
        # Since attention layer does't have weights, thus load the weight before adding the attention model
        # The return value "weights" is the timedistributed_1 layer's weight, we feed it to softmax function to make prediction
        weights = load_weights(rnn_model, self.weight_file_path)

        model = Sequential()
        model.add(rnn_model)

        # todo: check the source code of different simpleContextLayer
        if activation_rnn_size:
            print("debug: add SimpleContextLayer")
            model.add(SimpleContextLayer(self.config))
        # we are not going to fit so we dont care about loss and optimizer
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        wmodel = Sequential()
        wmodel.add(rnn_model)
        print("debug: add WSimpleContextLayer")
        wmodel.add(WSimpleContextLayer(self.config))
        wmodel.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model, wmodel, weights

    def score(self, samples):
        return self.keras_rnn_predict(samples)

    def keras_rnn_predict(self, samples):
        '''
        :param samples: left padded samples, demintion: (nb_sample, nb_word<25, vocab)
        :return: return the (vocab,) predicting distribution over the vaocabulary of next word given a sentence input
        '''
        """for every sample, calculate probability for every possible label
        you need to supply your RNN model and maxlen - the length of sequences it can handle
        """
        maxlend = self.config.maxlend
        eos = self.config.eos
        empty = self.config.empty
        maxlen = self.config.maxlen
        batch_size = self.config.batch_size

        sample_lengths = map(len, samples)
        assert all(l > maxlend for l in sample_lengths)
        assert all(l[maxlend] == eos for l in samples)
        # pad from right (post) so the first maxlend will be description followed by headline
        data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')

        # Make prediction
        # samples: (nb_sample, nb_word<25, vocab) E.g.: (1,18,2000)
        # prob[sample_length - maxlend - 1] is the last word(before padding) output of rnn
        # return the (vocab,) predicting distribution over the vaocabulary of next word given a sentence input
        if len(data) > 0:
            probs = self.model.predict(data, verbose=0, batch_size=batch_size)
        return np.array([self.output2probs(prob[sample_length - maxlend - 1]) for prob, sample_length in zip(probs, sample_lengths)])


    def to_word(self, x):
        return [self.idx2word[w] for w in x]


    # variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
    def beamsearch(self, score, start, avoid,  use_unk, temperature, k, avoid_score):
        """
        Beam search: each round random select k(10) candidates from the probabliltiy distribution
        Loop until all candidates are dead sentences which finished grow, return these sentences
        In each round, suppose 6 candidates out of 10 are growing sentences, let it grow.
        In each round, some growing sentence stopped grow

        All returned sentence starts with an `empty` label and end with `eos` or
        truncated to length of `maxsample`.

        You need to supply `predict` which returns the label probability of each sample.
        `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
        """
        maxlen = self.config.maxlen
        eos = self.config.eos
        empty = self.config.empty
        vocab_size = self.config.vocab_size
        maxsample = maxlen
        nb_unknown_words = self.config.nb_unknown_words
        beam_search_mode = self.config.beam_search_mode
        oov = range(vocab_size - nb_unknown_words, vocab_size)

        def sample_rank(energy, n, temperature, beam_search_mode = "random"):
            """sample at most n different elements according to their energy"""
            res = []
            n = min(n, len(energy))
            prb = np.exp(-np.array(energy) / temperature)

            for i in xrange(n):
                z = np.sum(prb)
                if beam_search_mode == "random":
                    r = np.argmax(np.random.multinomial(1, prb / z, 1))
                elif beam_search_mode == "hard":
                    r = np.argmax(prb)
                else:
                    raise ValueError('Unexpected values of beam_search_mode')
                res.append(r)
                prb[r] = 0.  # make sure we select each element only once
            return res



        dead_samples = []
        dead_scores = []
        live_samples = [list(start)] #start is a list of words
        live_scores = [0]


        while live_samples:
            print("live_samples")
            for x in live_samples:
                print(x)
                print(self.print_word_indexs(x))

            # for every possible live sample calc prob for every possible label
            # probs shape: (# live sample, vocab_size)
            probs = score(live_samples)
            # total score for every sample is sum of -log of word prb
            # The word empty doesn't include in the score calculation
            cand_scores = np.array(live_scores)[:,None] - np.log(probs)
            assert vocab_size == probs.shape[1]
            cand_scores[:,empty] = 1e20
            if not use_unk and oov is not None:
                cand_scores[:,oov] = 1e20
            if avoid:
                for a in avoid:
                    for i, s in enumerate(live_samples):
                        n = len(s) - len(start)
                        if n < len(a):
                            # at this point live_sample is before the new word,
                            # which should be avoided, is added
                            cand_scores[i,a[n]] += avoid_score
            live_scores = list(cand_scores.flatten())

            # find the best (lowest) scores we have from all possible dead samples and
            # all live samples and all possible new words added
            scores = dead_scores + live_scores

            # random select k best (word,score) from probablity scores distribution
            ranks = sample_rank(scores, k, temperature, beam_search_mode)

            n = len(dead_scores)
            # print("ranks", [(r - n) % vocab_size if r >= n else r for r in ranks])

            # Remove the those dead scores that are not in top k
            dead_scores = [dead_scores[r] for r in ranks if r < n]
            dead_samples = [dead_samples[r] for r in ranks if r < n]
            live_scores = [live_scores[r-n] for r in ranks if r >= n]
            # live_samples[(r-n)//vocab_size] is the coresponding sample
            # [(r-n)%vocab_size] is the predicting word
            # beam search: generate number of k-dead new live samples by adding k-dead new prediction to corresponding existing samples
            live_samples = [live_samples[(r-n)//vocab_size]+[(r-n)%vocab_size] for r in ranks if r >= n]
            # live samples that should be dead are...
            # even if len(live_samples) == maxsample we dont want it dead because we want one
            # last prediction out of it to reach a headline of maxlenh
            def is_zombie(s):
                '''
                :param s: sample with predictions at the end
                '''
                return s[-1] == eos or len(s) > maxsample

            # add zombies to the dead
            dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
            dead_samples += [s for s in live_samples if is_zombie(s)]

            # remove zombies from the living
            live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
            live_samples = [s for s in live_samples if not is_zombie(s)]

        return dead_samples, dead_scores

    def vocab_fold(self, xs):
        """convert list of word indexes that may contain words outside vocab_size to words inside.
        If a word is outside, try first to use glove_idx2idx to find a similar word inside.
        If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
        """
        vocab_size = self.config.vocab_size
        nb_unknown_words = self.config.nb_unknown_words
        glove_idx2idx = self.glove_idx2idx

        xs = [x if x < vocab_size - nb_unknown_words else glove_idx2idx.get(x, x) for x in xs]
        # the more popular word is <0> and so on
        outside = sorted([x for x in xs if x >= vocab_size - nb_unknown_words])
        # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
        outside = dict((x, vocab_size - 1 - min(i, nb_unknown_words - 1)) for i, x in enumerate(outside))
        xs = [outside.get(x, x) for x in xs]
        return xs

    def vocab_unfold(self, desc, xs):
        # assume desc is the unfolded version of the start of xs
        vocab_size = self.config.vocab_size
        nb_unknown_words = self.config.nb_unknown_words

        unfold = {}
        for i, unfold_idx in enumerate(desc):
            fold_idx = xs[i]
            if fold_idx >= vocab_size - nb_unknown_words:
                unfold[fold_idx] = unfold_idx
        return [unfold.get(x, x) for x in xs]

    def lpadd(self, x):
        """left (pre) pad a description to maxlend and then add eos.
        The eos is the input to predicting the first word in the headline
        """
        maxlend = self.config.maxlend
        eos = self.config.eos
        empty = self.config.empty


        assert maxlend >= 0
        if maxlend == 0:
            return [eos]
        n = len(x)
        if n > maxlend:
            x = x[-maxlend:]
            n = maxlend
        return [empty] * (maxlend - n) + x + [eos]


    def raw_to_index_in_vocab(self, x):
        word_indexes = []
        for w in x.split():
            w_clean = w.rstrip('^')
            if w_clean in self.word2idx:
                word_indexes.append(self.word2idx[w_clean])
            else:
                # x.append(config.unkonwn_word)
                # add unknow word to the last
                index = max(self.idx2word.keys(), key=int) + 1
                self.idx2word[index] = w_clean+"^"
                self.word2idx[w_clean+"^"] = index
                word_indexes.append(index)
        return word_indexes


    def print_word_indexs(self, x):
        return ' '.join(self.to_word(x))

    def print_samples(self, samples):
        for i, (y_predict, x_slice, score) in enumerate(samples):
            print i
            print("X slice:", self.print_word_indexs(x_slice))
            print("score: {}, Y predict:{}".format(score, self.print_word_indexs(y_predict)))


    def gensamples(self, X=None, X_test=None, Y_test=None, avoid=None, avoid_score=1, skips=2, k=10,
                   short=False, temperature=1., use_unk=True):
        '''
        :param X:
        :param X_test:
        :param Y_test:
        :param avoid:
        :param avoid_score:
        :param skips: Number of segmentation on X
        :param k: number of sample to generate for each X segmentation
        :param short: ignore similar sample output in print
        :param temperature:
        :param use_unk: alow to use unknonw place holder in prediction
        :return: sample list
        '''
        maxlend = self.config.maxlend
        eos = self.config.eos
        maxlen = self.config.maxlen
        empty = self.config.empty

        # Word indexing X
        if X is None or isinstance(X, int):
            if X is None:
                i = random.randint(0, len(X_test) - 1)
            else:
                i = X
            print 'HEAD %d:' % i, ' '.join(self.idx2word[w] for w in Y_test[i])
            print 'DESC:', ' '.join(self.idx2word[w] for w in X_test[i])
            sys.stdout.flush()
            x = X_test[i]
        else:
            # todo test this function
            temp = [self.word2idx[w.rstrip('^')] if w.rstrip('^') in self.word2idx else self.config.unkonwn_word
                    for w in X.split()]
            print("todo: test raw_to_index_in_vocab function")
            print("old raw to index, expecting the 'unknown' token for the new words")
            print(temp)
            # Raw words to indexes
            x = self.raw_to_index_in_vocab(X)
            print("raw_to_index_in_vocab(), expecting large index corresponding to 'unkonwn' token")
            print 'DESC:', ' '.join(self.idx2word[w] for w in x)
            print("x", x)
            print(self.print_word_indexs(x))


        # Avoid word
        if avoid:
            # avoid is a list of avoids. Each avoid is a string or list of word indeicies
            if isinstance(avoid, str) or isinstance(avoid[0], int):
                avoid = [avoid]
            avoid = [a.split() if isinstance(a, str) else a for a in avoid]
            avoid = [self.vocab_fold([w if isinstance(w, int) else self.word2idx[w] for w in a])
                     for a in avoid]

        # generate headline
        samples = []
        if maxlend == 0:
            skips = [0]
        else:
            skips = range(min(maxlend, len(x)), max(maxlend, len(x)), abs(maxlend - len(x)) // skips + 1)
        for s in skips:
            # after lpadd(), got random slice of x, which is x[s-maxlend:s]
            # start is the random slice of x
            # vocab fold x for predicting headline
            start = self.lpadd(x[:s])
            fold_start = self.vocab_fold(start)

            sample, score = self.beamsearch(score=self.keras_rnn_predict, start=fold_start, avoid=avoid,
                                       avoid_score=avoid_score,
                                       k=k, temperature=temperature, use_unk=use_unk)
            assert all(s[maxlend] == eos for s in sample)
            samples += [(s, start, scr) for s, scr in zip(sample, score)]


        # Sort sample based on score
        samples.sort(key=lambda x: x[-1])
        codes = []
        output = []
        for sample, start, score in samples:
            # vocab unfold sample using start
            # start is the unfold version of sample
            code = ''
            words = []
            # print("sample of indexes:", sample)
            sample = self.vocab_unfold(start, sample)[len(start):]
            # print("sample of indexes:", sample)
            # Get headline words and code for distance calculation
            for w in sample:
                if w == eos:
                    break
                words.append(self.idx2word[w].rstrip('^'))
                code += chr(w // (256 * 256)) + chr((w // 256) % 256) + chr(w % 256)
            output.append((score, ' '.join(words)))
            # print headline out
            # short version: Calculate edit distance between headline, won't output the one too close to each other
            if short:
                distance = min([100] + [-Levenshtein.jaro(code, c) for c in codes])
                # print("distance", distance)
                if distance > -0.6:
                    print score, ' '.join(words)
            # print '%s (%.2f) %f'%(' '.join(words), score, distance)
            else:
                print score, ' '.join(words)
            codes.append(code)

        # get matrix
        if len(samples) > 0:
            best_sample, start, _ = samples[0]
            data = sequence.pad_sequences([best_sample], maxlen=maxlen, value=empty, padding='post', truncating='post')
            weights = self.wmodel.predict(data, verbose=0, batch_size=1)
            print("data:", data)
            print("start:", start)
            unfold_sample = np.array(self.vocab_unfold(start, data[0, :]))
            print("unfold_sample", unfold_sample)

            startd = np.where(unfold_sample != empty)[0][0]
            lenh = np.where(unfold_sample[maxlend + 1:] == eos)[0][0]

            columns = [self.idx2word[unfold_sample[i]] for i in range(startd, maxlend)]
            rows = [self.idx2word[unfold_sample[i]] for i in range(maxlend + 1, maxlend + lenh + 1)]
            weights = weights[0, :lenh, startd:]

            dataframe = (weights, columns, rows)
            print("The length of output is:", output)
            print("top", k, "is:", output[:k])
            return output[:k], dataframe
        else:
            return None, None


class Predict_model2(Predict_model):
    def __init__(self, config, weight_file_path, word_embedding_file, language_model_path):
        super(Predict_model2, self).__init__(config, weight_file_path, word_embedding_file)
        self.language_model = kenlm.Model(language_model_path)

    def get_language_score(self, sentence):
        '''
        :param sentence: list of word
        :return:
        '''
        # language_model_weight = self.config.language_model_weight
        if sentence and len(sentence) > 0:
            score = -self.language_model.score(" ".join(sentence), bos=True, eos=True)#*language_model_weight
        else:
            score = 0
        return score

    def get_language_full_scores(self, sentence):
        '''
        :param sentence: list of word
        :return:
        '''
        if sentence and len(sentence) > 0:
            scores = self.language_model.full_scores(" ".join(sentence), bos=True, eos=True)
        else:
            scores = []
        return scores

    def sample_rank(self, energy, n, num_of_dead, temperature, dead_scores, dead_samples,
                    live_scores, live_samples, starting_point, beam_search_mode = "random"):
        """sample at most n different elements according to their energy"""
        vocab_size = self.config.vocab_size


        n = min(n, len(energy))
        prb = np.exp(-np.array(energy) / temperature)
        res = []
        for i in xrange(n * n):
            z = np.sum(prb)
            # np.random.multinomial(1, prb / z, 1) output: [0,0,0....,1,0,0..0]
            if beam_search_mode == "random":
                r = np.argmax(np.random.multinomial(1, prb / z, 1))
            elif beam_search_mode == "hard":
                r = np.argmax(prb)
            else:
                raise ValueError('Unexpected values of beam_search_mode')
            res.append(r)
            prb[r] = 0.  # make sure we select each element only once

        score = []
        for r in res:
            if r < num_of_dead:
                dead_score = dead_scores[r]
                dead_sample = self.to_word(dead_samples[r][starting_point:])
                l_score = self.get_language_score(dead_sample)
                score.append(dead_score + l_score)
                print("dead sample:", dead_sample)
                print([(w,s) for w, s in zip(dead_sample, self.get_language_full_scores(dead_sample))])
                print("dead_score: ", dead_score)
                print("language_score: ", l_score)
            else:
                live_score = live_scores[r - num_of_dead]
                live_sample = self.to_word(
                    live_samples[(r - num_of_dead) // vocab_size] + [(r - num_of_dead) % vocab_size])[starting_point:]
                l_score = self.get_language_score(live_sample)
                score.append(live_score + l_score)
                print("live sample:", live_sample)
                print([(w,s) for w, s in zip(live_sample, self.get_language_full_scores(live_sample))])
                print("live_score: ", live_score)
                print("language_score: ", l_score)
        print("total scores: ", sorted(zip(score, res)))
        res = [r for (s, r) in sorted(zip(score, res))][:n]
        print("top scores: ", res)
        return res

    # variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
    def beamsearch(self, score, start, avoid, use_unk, temperature, k, avoid_score):
        """
        Beam search: each round random select k(10) candidates from the probabliltiy distribution
        Loop until all candidates are dead sentences which finished grow, return these sentences
        In each round, suppose 6 candidates out of 10 are growing sentences, let it grow.
        In each round, some growing sentence stopped grow

        All returned sentence starts with an `empty` label and end with `eos` or
        truncated to length of `maxsample`.

        You need to supply `predict` which returns the label probability of each sample.
        `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
        """
        beam_search_mode = self.config.beam_search_mode
        maxlen = self.config.maxlen
        eos = self.config.eos
        empty = self.config.empty
        vocab_size = self.config.vocab_size
        maxsample = maxlen
        nb_unknown_words = self.config.nb_unknown_words
        oov = range(vocab_size - nb_unknown_words, vocab_size)

        dead_samples = []
        dead_scores = []
        live_samples = [list(start)]  # start is a list of words
        live_scores = [0]
        starting_point = len(list(start))

        while live_samples:
            # print("live_samples")
            # for x in live_samples:
            #     print(x)
            #     print(self.print_word_indexs(x))

            # for every possible live sample calc prob for every possible label
            # probs shape: (# live sample, vocab_size)
            probs = score(live_samples)
            # total score for every sample is sum of -log of word prb
            # The word empty doesn't include in the score calculation
            cand_scores = np.array(live_scores)[:, None] - np.log(probs)
            assert vocab_size == probs.shape[1]
            cand_scores[:, empty] = 1e20
            if not use_unk and oov is not None:
                cand_scores[:, oov] = 1e20
            if avoid:
                for a in avoid:
                    for i, s in enumerate(live_samples):
                        n = len(s) - len(start)
                        if n < len(a):
                            # at this point live_sample is before the new word,
                            # which should be avoided, is added
                            cand_scores[i, a[n]] += avoid_score
            live_scores = list(cand_scores.flatten())

            # find the best (lowest) scores we have from all possible dead samples and
            # all live samples and all possible new words added
            scores = dead_scores + live_scores

            n = len(dead_scores)
            # random select k best (word,score) from probablity scores distribution
            ranks = self.sample_rank(scores, k, n, temperature, dead_scores, dead_samples,
                    live_scores, live_samples, starting_point, beam_search_mode)


            # print("ranks", [(r - n) % vocab_size if r >= n else r for r in ranks])

            # Remove the those dead scores that are not in top k
            dead_scores = [dead_scores[r] for r in ranks if r < n]
            dead_samples = [dead_samples[r] for r in ranks if r < n]
            live_scores = [live_scores[r - n] for r in ranks if r >= n]
            # live_samples[(r-n)//vocab_size] is the coresponding sample
            # [(r-n)%vocab_size] is the predicting word
            # beam search: generate number of k-dead new live samples by adding k-dead new prediction to corresponding existing samples
            live_samples = [live_samples[(r - n) // vocab_size] + [(r - n) % vocab_size] for r in ranks if r >= n]

            # live samples that should be dead are...
            # even if len(live_samples) == maxsample we dont want it dead because we want one
            # last prediction out of it to reach a headline of maxlenh
            def is_zombie(s):
                '''
                :param s: sample with predictions at the end
                '''
                return s[-1] == eos or len(s) > maxsample

            # add zombies to the dead
            dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
            dead_samples += [s for s in live_samples if is_zombie(s)]

            # remove zombies from the living
            live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
            live_samples = [s for s in live_samples if not is_zombie(s)]

        return dead_samples, dead_scores


# API call in server init:
# input:
def load_model(config, weight_file_path, word_embedding_file):
    FN0 = config.FN0
    seed = config.seed
    prediction_model = config.prediction_model
    language_model_path = config.language_model_path


    # seed weight initialization
    random.seed(seed)
    np.random.seed(seed)

    word_embedding_file = '../sample_data/%s.pkl' % FN0
    if prediction_model == "Predic_model":
        predict_model = Predict_model(config, weight_file_path, word_embedding_file)
    elif prediction_model == "Predic_model2":
        predict_model = Predict_model2(config, weight_file_path, word_embedding_file, language_model_path)
    return predict_model



def main():
    # setup configurations
    config = Choose_config.current_config['class']()
    FN0 = config.FN0
    MODEL_PATH = config.MODEL_PATH
    DATA_PATH = config.DATA_PATH
    config_name = Choose_config.current_config['name']
    beam_search_size= config.beam_search_size
    _, weight_file_path = get_weight_file(MODEL_PATH, config_name)
    word_embedding_file = DATA_PATH + '%s.pkl' % FN0
    seed = config.seed

    # seed weight initialization
    random.seed(seed)
    np.random.seed(seed)

    # Load model
    predict_model = load_model(config=config, weight_file_path=weight_file_path, word_embedding_file=word_embedding_file)


    print("Start predict test 1")
    # val_path = "../sample_data/test.jsonl"
    # val_sample_gen = load_article(raw_path=val_path, early_stop=80, batch_size=4)
    # article = val_sample_gen.next()
    # X = article['content'][0]
    # Y = article['title'][0]()
    # print("X:", X)
    # print("Y:", Y)
    # while(True):
    #     print()
    #     X = raw_input()

    start = time.time()
    X = "President Barack Obama 's re-election campaign is fundraising off of comments on Obama 's birth certificate by Mitt Romney 's son Matt ."
    print("X:", X)
    samples = predict_model.gensamples(X=X, skips=2, k=beam_search_size, temperature=1.,use_unk=False)
    end = time.time()
    print("time took:", end - start)
    #
    # X = "Russia's President Vladimir Putin has condemned Turkey's shooting down of a Russian warplane on its border with Syria."
    # print("X:", X)
    # samples = predict_model.gensamples(X=X, skips=2, k=10, temperature=1.,use_unk=False)
    # X = '''When Sarah Palin defended governors who are refusing to accept refugees by claiming there's no vetting process to keep out terrorists, "Late Night" host Seth Meyers completely shut down her argument.'''
    # print("X:", X)
    # samples = predict_model.gensamples(X=X, skips=2, k=10, temperature=1.,use_unk=False)



    # print("Start predict test 2")
    #
    # # Load configs
    # maxlen = config.maxlen
    # empty = config.empty
    # context_weight = K.variable(1.)
    # head_weight = K.variable(1.)
    # context_weight.set_value(np.float32(1.))
    # head_weight.set_value(np.float32(1.))
    #
    # X = "Representatives of the groups depicted in The Revenant^ spoke with BuzzFeed News about the actor 's Golden Globes speech calling on listeners to `` protect ... indigenous lands . ''"
    # Y = "Native American Groups Officially Respond To Leonardo DiCaprio 's Call To Action"
    # print("X:", X)
    # print("Y:", Y)
    # samples = predict_model.gensamples(X, skips=2, k=10, temperature=1.)




if __name__ == "__main__":
    main()



