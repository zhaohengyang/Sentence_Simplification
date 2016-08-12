from keras.layers.core import Lambda
import keras.backend as K


class simple_context_wrapper():
    def __init__(self, config):
        self.config = config

    def simple_context_fun(self, X, mask):
        n, maxlend, maxlenh = self.config.activation_rnn_size, self.config.maxlend, self.config.maxlenh
        desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
        head_activations, head_words = head[:,:,:n], head[:,:,n:]
        desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]

        # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
        # activation for every head word and every desc word
        # head_activations is (64, 25, 40) and desc_activations is (64, 25, 40)
        # head_activations is ([batch0-batch63], [head0-head24], [feature0-feature39])
        # desc_activations is ([batch0-batch63], [desc0-desc24], [feature0-feature39])
        # activation_energies is ([batch0-batch63], [Score(head0, all_desc)-Score(head24, all_desc)], [Score(headi, desc0)-Score(headi, desc24)])
        # note: score function is the similarity between two vector here.
        activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2)) # (64,25,25)
        # make sure we dont use description words that are masked out
        activation_energies = activation_energies + -1e20 * K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)

        # for every head word compute weights for every desc word
        activation_energies = K.reshape(activation_energies,(-1,maxlend)) # (64*25,25)
        # note: only for convenience by doing p(y=y0|64 batch Xs) = softmax([64*25]) instead of p(y = y0|x) = softmax([x0...x24])
        activation_weights = K.softmax(activation_energies) # (64*25,25)
        activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend)) # (64,25,25)

        # for every head word compute weighted average of desc words
        # ativation_weights, desc_words are: (64,25,25), (64,25,512-40)
        # activation_energies is ([batch0-batch63], [Score(head0, all_desc)-Score(head24, all_desc)], [Score(headi, desc0)-Similarity(headi, desc24)])
        # desc_words is ([batch0-batch63], [desc0-desc24], [feature40-feature512])
        # batch_dot: sum over 25 descpritions: [[Score(h0,d0)*d0 + Score(h0,d1)*d1 + ... + Score(h0,d24)*d24]
        #                                       [Score(h1,d0)*d0 + Score(h1,d1)*d1 + ... + Score(h1,d24)*d24]
        #                                       ...
        #                                       [Score(h24,d0)*d0 + Score(h24,d1)*d1 + ... + Score(h24,d24)*d24]]
        # Which is equal to [c0, c24]
        desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1)) # (64,25,472)
        # for each head_t, the feature is [c_t; h_t]
        return K.concatenate((desc_avg_word, head_words)) # (64,25,944) where head_words is (64, 25, 472)

    def wsimple_context_fun(self, X, mask):
        n, maxlend, maxlenh = self.config.activation_rnn_size, self.config.maxlend, self.config.maxlenh
        # the numpy slicing is not working in tensorflow backend
        desc, head = X[:,:maxlend], X[:,maxlend:]
        head_activations, head_words = head[:,:,:n], head[:,:,n:]
        desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]

        # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
        # activation for every head word and every desc word
        activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2],[2]))
        # make sure we dont use description words that are masked out
        assert mask.ndim == 2
        activation_energies = K.switch(mask[:, None, :maxlend], activation_energies, -1e20)

        # for every head word compute weights for every desc word
        activation_energies = K.reshape(activation_energies,(-1,maxlend))
        activation_weights = K.softmax(activation_energies)
        activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

        return activation_weights

class SimpleContextLayer(Lambda):
    def __init__(self, config):
        self.config = config
        wrapper = simple_context_wrapper(self.config)

        super(SimpleContextLayer, self).__init__(wrapper.simple_context_fun)
        self.supports_masking = True


    def compute_mask(self, input, input_mask=None):
        maxlend = self.config.maxlend
        return input_mask[:, maxlend:]

    def get_output_shape_for(self, input_shape):
        rnn_size = self.config.rnn_size
        activation_rnn_size = self.config.activation_rnn_size
        maxlenh = self.config.maxlenh

        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)




class WSimpleContextLayer(Lambda):
    def __init__(self, config):
        self.config = config
        wrapper = simple_context_wrapper(self.config)

        super(WSimpleContextLayer, self).__init__(wrapper.wsimple_context_fun)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        maxlend = self.config.maxlend
        return input_mask[:, maxlend:]

    def get_output_shape_for(self, input_shape):
        rnn_size = self.config.rnn_size
        activation_rnn_size = self.config.activation_rnn_size
        maxlenh = self.config.maxlenh

        nb_samples = input_shape[0]
        n = 2*(rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)


