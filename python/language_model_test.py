from time import time
import kenlm
import sys

"""
usage: python language_model_test.py ../../lib/kenlm/models/ngram_lm.trie
"""
def main():
    if len(sys.argv) < 2:
        print("Usage: give path parameter")
    else:
        _, model_path = sys.argv
        model = kenlm.Model(model_path)

        input = ''
        while(input != 'quit'):
            print("Type a sentence to get a score: (quit)")
            input = raw_input()
            start = time()
            print("input:",input)
            print(model.score(input, bos=True, eos=True))
            time_took = time() - start
            print("time took:", time_took)


if __name__ == '__main__':
    main()