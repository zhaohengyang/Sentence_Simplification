from headline_generator.config import Config1, Config2, Choose_config
from headline_generator.model import model1
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
from sklearn.cross_validation import train_test_split
import pickle
# from keras.utils.visualize_util import plot
from keras.callbacks import Callback
import keras.backend as K
from headline_generator.data import load_article
import fnmatch
import os
import re



class Data_generator():
    def __init__(self, idx2word, glove_idx2idx, config):
        self.config = config
        self.glove_idx2idx = glove_idx2idx
        self.idx2word = idx2word

    def prt(self, label, x):
        idx2word = self.idx2word
        print label + ':',
        for w in x:
            print idx2word[w],
        print

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
        return [empty]*(maxlend-n) + x + [eos]

    def flip_headline(self, x, nflips=None, model=None, debug=False):
        """given a vectorized input (after `pad_sequences`) flip some of the words in the second half (headline)
        with words predicted by the model
        """
        # Load settings
        maxlend = self.config.maxlend
        maxlen = self.config.maxlen
        eos = self.config.eos
        empty = self.config.empty
        oov0 = self.config.oov0
        idx2word = self.idx2word

        if nflips is None or model is None or nflips <= 0:
            return x

        batch_size = len(x)
        assert np.all(x[:, maxlend] == eos)
        probs = model.predict(x, verbose=0, batch_size=batch_size)
        x_out = x.copy()
        for b in range(batch_size):
            # pick locations we want to flip
            # 0...maxlend-1 are descriptions and should be fixed
            # maxlend is eos and should be fixed
            flips = sorted(random.sample(xrange(maxlend + 1, maxlen), nflips))
            if debug and b < debug:
                print b,
            for input_idx in flips:
                if x[b, input_idx] == empty or x[b, input_idx] == eos:
                    continue
                # convert from input location to label location
                # the output at maxlend (when input is eos) is feed as input at maxlend+1
                label_idx = input_idx - (maxlend + 1)
                prob = probs[b, label_idx]
                w = prob.argmax()
                if w == empty:  # replace accidental empty with oov
                    w = oov0
                if debug and b < debug:
                    print '%s => %s' % (idx2word[x_out[b, input_idx]], idx2word[w]),
                x_out[b, input_idx] = w
            if debug and b < debug:
                print
        return x_out


    def vocab_fold(self, xs):
        """convert list of word indexes that may contain words outside vocab_size to words inside.
        If a word is outside, try first to use glove_idx2idx to find a similar word inside.
        If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
        """
        oov0 = self.config.oov0
        vocab_size = self.config.vocab_size
        nb_unknown_words = self.config.nb_unknown_words
        glove_idx2idx = self.glove_idx2idx


        xs = [x if x < oov0 else glove_idx2idx.get(x, x) for x in xs]
        # the more popular word is <0> and so on
        outside = sorted([x for x in xs if x >= oov0])
        # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
        outside = dict((x, vocab_size - 1 - min(i, nb_unknown_words - 1)) for i, x in enumerate(outside))
        xs = [outside.get(x, x) for x in xs]
        return xs

    def conv_seq_labels(self, xds, xhs, nflips=None, model=None, debug=False):
        """description and hedlines are converted to padded input vectors. headlines are one-hot to label"""
        maxlen = self.config.maxlen
        maxlenh = self.config.maxlenh
        vocab_size = self.config.vocab_size
        empty = self.config.empty
        eos = self.config.eos

        batch_size = len(xhs)
        assert len(xds) == batch_size
        # pad input to same size: [empty]...[empty] Example description [eos] Example headline [empty]...[empty]
        if debug:
            self.prt('D cutted', xds[0])

        # fold x(In large vocab) into word in vocab and 100 place holders
        x = [self.vocab_fold(self.lpadd(xd) + xh) for xd, xh in zip(xds, xhs)]  # the input does not have 2nd eos
        x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')

        if debug:
            self.prt('D pad', x[0])
            print("x[0] {}".format(x[0]))

        # flip some data from xh to model prediction
        x = self.flip_headline(x, nflips=nflips, model=model, debug=debug)

        #     print("x {}".format(x))
        y = np.zeros((batch_size, maxlenh, vocab_size))

        if debug:
            self.prt("H cutted", xhs[0])

        for i, xh in enumerate(xhs):

            # right padding
            # xh append [eos] and 25 [empty] then cut the end: Example: This yeild a great outcome [eos] [empty1] [empty19]
            xh = self.vocab_fold(xh) + [eos] + [empty] * maxlenh  # output does have a eos at end
            xh = xh[:maxlenh]

            if debug:
                if i ==0:
                    self.prt("H pad", xh)

            # change xh to one-hot matrix, each item in xh become a one-hot vector
            y[i, :, :] = np_utils.to_categorical(xh, vocab_size)

        return x, y


    def gen(self, article_gen, word2idx, nb_batches = None, nflips=None, model=None, debug=False):
        """yield batches.
           while training it is good idea to flip once in a while the values of the headlines from the
           value taken from Xh to value generated by the model.
        """
        batch_size = self.config.batch_size
        maxlend = self.config.maxlend
        maxlenh = self.config.maxlenh
        seed = self.config.seed
        c = nb_batches if nb_batches else 0
        while True:
            for articles in article_gen():
                if nb_batches and c >= nb_batches:
                    c = 0
                X_raw = [[word2idx[token] for token in d.split()] for d in articles['content']]
                Y_raw = [[word2idx[token] for token in headline.split()] for headline in articles['title']]

                assert len(X_raw) == len(Y_raw)
                # Random slice the x and y
                new_seed = random.randint(0, sys.maxint)
                random.seed(c + 123456789 + seed)
                for i in range(len(X_raw)):
                    s = random.randint(min(maxlend, len(X_raw[i])), max(maxlend, len(X_raw[i])))
                    X_raw[i] = X_raw[i][:s]

                    s = random.randint(min(maxlenh, len(Y_raw[i])), max(maxlenh, len(Y_raw[i])))
                    Y_raw[i] = Y_raw[i][:s]

                # undo the seeding before we yield inorder not to affect the caller
                c += 1
                random.seed(new_seed)

                # Padding
                x, y = self.conv_seq_labels(X_raw, Y_raw, nflips=nflips, model=model, debug=debug)
                yield (x, y)



    # def gen(self, Xd, Xh, nb_batches=None, nflips=None, model=None, debug=False):
    #     """yield batches. for training use nb_batches=None
    #     for validation generate deterministic results repeating every nb_batches
    #
    #     while training it is good idea to flip once in a while the values of the headlines from the
    #     value taken from Xh to value generated by the model.
    #     """
    #     #     print("len(Xd)", len(Xd))
    #     batch_size = self.config.batch_size
    #     maxlend = self.config.maxlend
    #     maxlenh = self.config.maxlenh
    #     seed = self.config.seed
    #
    #     c = nb_batches if nb_batches else 0
    #     while True:
    #         xds = []
    #         xhs = []
    #         if nb_batches and c >= nb_batches:
    #             c = 0
    #         new_seed = random.randint(0, sys.maxint)
    #         random.seed(c + 123456789 + seed)
    #         for b in range(batch_size):
    #             t = random.randint(0, len(Xd) - 1)
    #
    #             # Cut xd and xh, in order to generate different text from same input X_i
    #             xd = Xd[t]
    #
    #             #             print("maxlend: {}, len(xd): {}".format(maxlend, len(xd)))
    #             s = random.randint(min(maxlend, len(xd)), max(maxlend, len(xd)))
    #             #             print("xd[:s]: {}".format(xd[:s]))
    #
    #             xds.append(xd[:s])
    #
    #             xh = Xh[t]
    #             s = random.randint(min(maxlenh, len(xh)), max(maxlenh, len(xh)))
    #             xhs.append(xh[:s])
    #
    #         #             print("maxlend: {}, len(xd): {}".format(maxlenh, len(xh)))
    #         #             print("len(xd[:s]) {}, len(xh[:s]) {}".format(len(xd[:s]), len(xh[:s])))
    #
    #         # undo the seeding before we yield inorder not to affect the caller
    #         c += 1
    #         random.seed(new_seed)
    #
    #         x, y = self.conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)
    #         #         print("x.shape {}, y.shape {}".format(x.shape, y.shape))
    #         yield x, y

    def test_gen(self, gen, config, n=3):
        maxlend = config.maxlend
        eos = config.eos

        for i in range(2):
            print("interation:", i)
            Xtr,Ytr = next(gen)
            print(Xtr.shape, Ytr.shape)
            for i in range(n):
                assert Xtr[i,maxlend] == eos
                x = Xtr[i,:maxlend]
                y = Xtr[i,maxlend:]
                yy = Ytr[i,:]
                # From one hot matrix to index vector
                yy = np.where(yy)[1]
                self.prt('L',yy)
                self.prt('H',y)
                if maxlend:
                    self.prt('D',x)



def get_weight_file(weight_folder, config_name):
    '''
    input: model folder like: '/Users/zhaohengyang/PycharmProjects/FoxType_mike/model/'
    :return: weight file with highest repo
    '''
    pattern = config_name + '.epoch_*.hdf5'
    p = re.compile(config_name + ".epoch_(.*).hdf5")
    weight_files = []
    for root, dirs, files in os.walk(weight_folder):
        for filename in fnmatch.filter(files, pattern):
            epoch = int(p.search(filename).group(1))
            weight_files += [(epoch, os.path.join(root, filename))]
    weight_files.sort(key=lambda x: x[0], reverse=True)
    return weight_files[0]

def main(debug=False):
    # Load configuration
    config_name = Choose_config.current_config['name']
    config = Choose_config.current_config['class']()
    batch_size = config.batch_size
    seed = config.seed
    FN0 = config.FN0
    FN1 = config.FN1
    nflips = config.nflips
    nb_epoch = config.nb_epoch
    LR = config.LR
    early_stop = config.early_stop
    nb_val_samples = config.nb_val_samples
    train_path = "../sample_data/train.jsonl"
    val_path = "../sample_data/test.jsonl"
    train_on_weight = config.train_on_weight

    # load word embeddings
    with open('../sample_data/%s.pkl' % FN0, 'rb') as fp:
        embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)

    print("word embedding shape", embedding.shape)
    # load model sturcture
    model = config.model['class'](embedding, config)
    plot(model, to_file='model.png', show_shapes=True)

    # load model weight
    if train_on_weight:
        config_name = Choose_config.current_config['name']
        weight_folder = config.MODEL_PATH
        newest_epoch, init_trained_weight_path = get_weight_file(weight_folder, config_name)
        start_epoch = newest_epoch + 1
        print("start epo:", start_epoch)
        print("trained on weights: " + init_trained_weight_path)
        model.load_weights(init_trained_weight_path)


    # generate simulate data
    load_train = lambda: load_article(raw_path=train_path, early_stop=early_stop, batch_size=batch_size)
    load_test = lambda: load_article(raw_path=val_path, early_stop=nb_val_samples, batch_size=batch_size)
    data_generator = Data_generator(idx2word, glove_idx2idx, config)
    traingen = data_generator.gen(load_train, word2idx, nflips=nflips, model=model)
    valgen = data_generator.gen(load_test, word2idx)

    # test generator
    if debug:
        data_generator.test_gen(traingen, config)



    # train
    history = {}
    for iteration in range(nb_epoch):
        print 'Iteration', iteration
        # Assume train sample size is 1000, each time is will feed 64(batch_size) sample to train,
        # it will switch to next epoch when the 64*batch_round is exceeding 1000
        h = model.fit_generator(traingen, samples_per_epoch=early_stop,
                                nb_epoch=1#, validation_data=valgen, nb_val_samples=nb_val_samples
                                )

        # append new h.history to history list
        for k, v in h.history.iteritems():
            history[k] = history.get(k, []) + v

        with open('../model/%s.history.pkl' % FN1, 'wb') as fp:
            pickle.dump(history, fp, -1)
        # if iteration % 5 == 0:
        model_weight_path = '../model/{}.epoch_{}.hdf5'.format(config_name, iteration + start_epoch)
        model.save_weights(model_weight_path, overwrite=True)

        if iteration > 5:
            # Reduce learning rate each epoch
            LR *= 0.5
            K.set_value(model.optimizer.lr, np.float32(LR))
if __name__ == "__main__":
    main(debug=False)

