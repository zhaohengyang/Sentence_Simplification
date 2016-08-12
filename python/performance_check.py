import pickle
from headline_generator.config import Choose_config, Config1, Config2

def plot_model_performance(result):
    # Compare models' accuracy, loss and elapsed time per epoch.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 10))
    plt.style.use('ggplot')
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax2.set_title('Losses')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')

    x = range(len(result['loss']))
    ax2.plot(x, result['val_loss'], label='val_loss')
    ax2.plot(x, result['loss'], label='loss')

    ax2.legend()
    plt.tight_layout()
    plt.show()

def main():
    config = Choose_config().current_config['class']()
    FN1 = config.FN1
    print(FN1)
    result = pickle.load(open('../model/%s.history.pkl' % FN1, 'rb'))
    print(result)
    # plot_model_performance(result)




if __name__ == "__main__":
    main()

