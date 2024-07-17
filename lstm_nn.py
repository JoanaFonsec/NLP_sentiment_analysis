from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from utils import confusion_matrix


class LSTM_NN:
    def __init__(self):
        """ Initialize the LSTM (Long short term memory) Neural Network model
            and the tokenizer that transforms words into int vectors.
        """
        # PREPARE WORD EMBEDDING
        vocab_size = 4000
        self.tokenizer = Tokenizer(num_words=vocab_size)

        # DEFINE MODEL ARCHITECTURE
        self.model = Sequential([
            Embedding(vocab_size, 100, input_length=50),
            Bidirectional(LSTM(64)),
            Dense(24, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # COMPILE MODEL
        # self.model.build(input_shape=(None, 50))
        self.model.compile(optimizer="adam", loss="binary_crossentropy",
                           metrics=["accuracy"])
        # self.model.summary()

    def performance(self, xtrain, xvalid, ytrain, yvalid):
        """
        Trains Neural Network model using training dataset.

        Args:
            xtrain (list of strings): list of processed tweets for training
            xvalid (list of strings): list of processed tweets for validation
            ytrain (list of ints): list of labels containing the sentiment
                                    {0: Negative, 1: Positive} for training
            yvalid (list of ints): list of labels containing the sentiment
                                    {0: Negative, 1: Positive} for validation

        Returns:
            float: Accuracy of the model using the validation dataset
        """
        # EMBED WORDS AS INT VECTORS
        self.tokenizer.fit_on_texts(xtrain)
        train_bow = self.tokenizer.texts_to_sequences(xtrain)
        xtrain = pad_sequences(train_bow, padding='post', maxlen=30)

        valid_bow = self.tokenizer.texts_to_sequences(xvalid)
        xvalid = pad_sequences(valid_bow, padding='post', maxlen=30)

        # FIT MODEL WEIGHTS TO TRAINING DATA
        self.model.fit(xtrain, ytrain, epochs=1, batch_size=32,
                       validation_data=(xvalid, yvalid), verbose=1)

        # EVALUATE MODEL ACCURACY
        predicted = (self.model.predict(xvalid) > 0.5).astype("int32")

        return confusion_matrix(yvalid, predicted)
