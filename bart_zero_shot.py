import torch
from transformers import BartForSequenceClassification, BartTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class BartZeroShot:
    def __init__(self):
        """ Initialize the BART (Bidirectional and Auto-Regressive Transformer)
            model and the tokenizer that transforms words into int vectors.
        """
        self.nli_model = BartForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli")
        self.tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-large-mnli")

    def predict(self, sentence, label):
        """ Predict the likelihood that a sentence
            expresses a sentiment given by a label

        Args:
            sentence (string): Sentence to predict the labeled sentiment
            label (string): Type of sentiment {Positive, Negative, Neutral,...}

        Returns:
            float: Probability that the sentence conveys the labeled sentiment
        """

        # ENCODE SENTENCE AND LABEL
        x = self.tokenizer.encode(
            sentence,
            f"This example is {label}",
            return_tensors="pt",
            truncation="only_first",
        )
        logits = self.nli_model(x.to(DEVICE))[0]

        # GET LIKELIHOOD
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(1)
        prob_label_is_true = probs[:, 1].item()

        return prob_label_is_true

    def performance(self, file):
        """ Predict and calculate the performance of the BART model
            for the data in a file

        Args:
            file (.csv): File containing all the data

        Returns:
            float: Accuracy of the model
        """

        # INIT VARIABLES & MODEL EVALUATION
        ncorrect, nsamples = 0, 0
        self.nli_model.eval()

        # READ FILE, GET SENTENCE AND LABEL
        with open(file) as fin:
            fin.readline()
            for i, line in enumerate(fin):
                tokens = line.strip().split(',')
                sent, target = tokens[4:-1], tokens[0]

                # CALCULATE PREDICTION
                prediction = self.predict(sent, "positive")
                label = prediction >= 0.5
                ncorrect += int(label == target)
                nsamples += 1

        return float(ncorrect)/float(nsamples)
