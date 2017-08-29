# char-rnn Trump-like text generation

Keras 2.x (tensorflow backend) char-rnn trained with the Trump Twitter Archive (http://www.trumptwitterarchive.com/)

Two python notebooks, one for training and one for testing.

You can use the dataset, train a model from scratch, or skip that part and use the provided weights to play with the text generation (have fun!).

Note: The training code is multi-GPU compatible. From my experience, you can significantly reduce training time if the sequences you feed the network are long (120 for example). With 2-GPUs, I got -56% time per epoch.

