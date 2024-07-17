from textblob.classifiers import NaiveBayesClassifier as NBC
from utils import confusion_matrix


class NaiveBayes:
    def __init__(self):
        """
        Create Naive Bayes model
        """
        self.model = None

    def performance(self, xtrain, xvalid, ytrain, yvalid):
        """
        Trains Naive Bayes model using training dataset.

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

        # MERGE DATA AND CLASSIFIER
        train_list = [(xtrain[i], ytrain[i]) for i in range(0, len(xtrain))]

        # TRAIN AND FIT MODEL
        self.model = NBC(train_list)

        predicted = []
        for sentence in xvalid:
            predicted.append(self.model.classify(sentence))

        return confusion_matrix(yvalid, predicted)
