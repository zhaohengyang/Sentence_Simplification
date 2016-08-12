from __future__ import unicode_literals, print_function, division
from headline_generator.config import Config1, Config2, Choose_config
from headline_generator.predict import load_model
from headline_generator.train import get_weight_file
import os
import logging
import pandas as pd
import json
from nltk import word_tokenize

from flask import Flask, jsonify, request
from flask.ext.cors import CORS

app = Flask(__name__)
cors = CORS(app)

# Todo: test api
print("Headline generation server start:")
# Setup configurations
config = Choose_config.current_config['class']()
FN0 = config.FN0
MODEL_PATH = config.MODEL_PATH
DATA_PATH = config.DATA_PATH
config_name = Choose_config.current_config['name']
_, weight_file_path = get_weight_file(MODEL_PATH, config_name)
word_embedding_file = DATA_PATH + '%s.pkl' % FN0
beam_search_size = config.beam_search_size
print("Loading weight file:", weight_file_path)
print("Load Model...")
# Load model
predict_model = load_model(config = config, weight_file_path=weight_file_path, word_embedding_file=word_embedding_file)

print("Server ready for request")

# Type this for test server code
#import requests
#requests.get('http://52.201.240.247:8000/headline_generator', params={'sentence': 'how are you?'})
@app.route('/headline_generator', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        response = request.get_json(force=True)
        sentence = response.get('sentence')
        output, dataframe = predict_model.gensamples(X=sentence, skips=2, k=beam_search_size, temperature=1., use_unk=False)
        if output:
            (weights, columns, rows) = dataframe

            print("headlines:", output)
            print("weights.shape:", weights.shape)
            df = pd.DataFrame(weights, columns=columns, index=rows)
            print(df.head(20))
            return jsonify({
                'top_five_headlines': output,
                'weight_matrix': {"articles": columns,
                                  "top_headline": rows,
                                  "weights": weights.tolist()
                                  }
            })
            output = [dict({"score": score, "sentence": sentence}) for score, sentence in output]

        else:
            return "No headline", 400
    else:
        sentence = request.args.get('sentence')
        if not sentence:
            return "No sentence", 400
        output, dataframe = predict_model.gensamples(X=sentence, skips=2, k=beam_search_size, temperature=1., use_unk=False)
        if output:
            (weights, columns, rows) = dataframe

            print("headlines:", output)
            print("weights.shape:", weights.shape)
            df = pd.DataFrame(weights, columns=columns, index=rows)
            print(df.head(20))
            output = [dict({"score": score, "sentence": sentence}) for score, sentence in output]

            return jsonify({
                'top_five_headlines': output,
                'weight_matrix': {"articles": columns,
                                  "top_headline": rows,
                                  "weights": weights.tolist()
                                  }
            })
        #json.dumps
        else:
            return "No headline", 400


@app.route('/get_example', methods=['GET'])
def hello_get():
    sentence = request.args.get('sentence')
    if not sentence:
        return "No sentence", 400

    sentence = "hello"
    return jsonify({
        'original': sentence,
    })


@app.route('/post_example', methods=['POST'])
def hello_post():
    response = request.get_json(force=True)
    sentence = response.get('sentence')

    return jsonify({'sentence': sentence})

@app.before_first_request
def setup_logging():
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.addHandler(logging.StreamHandler())
        app.logger.setLevel(logging.INFO)


@app.route('/ping', methods=['GET'])
def hello_check():
    return 'pong', 200


@app.route('/fail', methods=['GET'])
def hello_fail():
    raise ValueError("Expected fail.")


if __name__ == '__main__':
    app.debug = True if os.environ.get('APP_DEBUG') else False
    app.profile = True if os.environ.get('APP_PROFILE') else False
    print("app.debug", app.debug)
    port = os.environ.get('PORT') or 8000
    print('port ' + str(port))
    app.run(port=int(port), use_reloader=False, host='0.0.0.0', threaded=True)

