from headline_generator.model import model1

class Config1():
    # path:
    ROOT_PATH = "../"
    MODEL_PATH = ROOT_PATH + "model/"
    DATA_PATH = ROOT_PATH + "smaple_data/"

    # Preprocessing
    early_stop = 100
    # word embedding
    glove_word_embedding_300 = {"glove_name": "../model/glove_word_embedding/glove.6B/glove.6B.300d.txt",
                                "embedding_dim": 300,
                                "glove_thr": 0.5
                                }
    vocab_size = 2000 # has to be smaller than len(vocab)

    # Special vocabulary symbols
    _PAD = b"_PAD"
    _GO = b"_GO"
    _EOS = b"_EOS"
    _UNK = b"_UNK"
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]

    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3


    # padding parameters
    maxlend = 25  # 0 - if we dont want to use description at all
    maxlenh = 25
    maxlen = maxlend + maxlenh

    # model sturcture parameters
    rnn_size = 512  # must be same as 160330-word-gen
    rnn_layers = 3
    batch_norm = False
    activation_rnn_size = 40 if maxlend else 0

    # training parameters
    seed = 42
    p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
    optimizer = 'adam'
    LR = 1e-4
    batch_size = 64
    nflips = 10
    nb_epoch = 5
    nb_val_samples = 1000
    train_on_weight = True

    # Prediction parameters
    prediction_model_selection = {"Predic_model": "Predic_model", "Predic_model2": "Predic_model2"}
    beam_search_mode_selection = {"random":"random", "hard":"hard"}
    language_model_path = "../../lib/kenlm/models/ngram_lm.trie"
    prediction_model = prediction_model_selection["Predic_model2"]
    beam_search_mode = beam_search_mode_selection["hard"]
    # language_model_weight = 2
    # phrase parameters
    FN0 = 'vocabulary-embedding'
    FN1 = 'train'

    # Model structure selection
    model = {"class":model1}


class Config2(Config1):
    LR = 1e-4
    vocab_size = 40000
    oov0 = vocab_size - Config1.nb_unknown_words
    early_stop = 999000 # Load the whole train data
    nb_epoch = 200
    batch_size = 64

class Choose_config():
    current_config = {"name":"Config2", "class":Config2}




