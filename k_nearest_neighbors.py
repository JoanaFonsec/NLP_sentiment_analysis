from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from utils import confusion_matrix


class KNearestNeighbors:
    def __init__(self):
        """
        Create K-Nearest Neighbors model and vectorizer
        """
        self.model = KNeighborsClassifier(n_neighbors=50, weights='distance')
        self.vectorizer = TfidfVectorizer(max_features=500)

    def performance(self, xtrain, xvalid, ytrain, yvalid):
        """
        Trains K-Nearest Neighbors model using training dataset.

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

        # BAG OF WORDS
        xtrain = self.vectorizer.fit_transform(xtrain)
        xvalid = self.vectorizer.fit_transform(xvalid)

        # TRAIN MODEL
        self.model.fit(xtrain, ytrain)
        predicted = self.model.predict(xvalid)

        return confusion_matrix(yvalid, predicted)
