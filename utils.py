from preprocess import cleanup
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def get_data(file, random_state=42):
    """Extracts and separates the data from the file.

    Args:
        file (.csv): File with all the input data
        random_state (int, optional): Key to generate "random" divisions of
                            the training and validation sets. Defaults to 42.

    Returns:
        tuple: Contains the training and validation data sets and labels
    """
    listData = []
    listLabel = []

    # GET DATA AND LABEL
    with open(file) as fin:
        fin.readline()
        for i, line in enumerate(fin):
            tokens = line.strip().split(',')

            # CHECK FOR EMPTY LINES
            if len(tokens) < 3:
                continue

            tweet = ','.join(tokens[4:])
            listData.append(''.join(tweet))
            listLabel.append(int(tokens[0]))

    # SPLIT DATA IN TRAIN AND VALIDATE SETS
    xtrain, xvalid, ytrain, yvalid = train_test_split(
        listData, listLabel, random_state=random_state, test_size=0.3)

    # REMOVE BART PREDICTIONS FROM TRAINING SET
    new_xtrain = []
    for line in xtrain:
        tokens = line.strip().split(',')
        tweet = ','.join(tokens[:-1])
        tweet = cleanup(tweet)
        new_xtrain.append(' '.join(tweet))

    # REMOVE BART PREDICTIONS FROM VALIDATION SET
    new_xvalid = []
    ncorrect = 0
    nsamples = 0
    for line in xvalid:
        tokens = line.strip().split(',')
        tweet = ','.join(tokens[:-1])
        tweet = cleanup(tweet)
        new_xvalid.append(' '.join(tweet))

        # SAVE BART PREDICTIONS FOR ACCURACY
        prediction = float(tokens[-1]) >= 0.5
        ncorrect += int(int(prediction) == int(yvalid[nsamples]))
        nsamples += 1

    # GET ACCURACY FOR VALIDATION SET
    ans = float(ncorrect)/float(nsamples)

    # FORMAT AS NP ARRAY
    ytrain = np.asarray(ytrain)
    yvalid = np.asarray(yvalid)

    return new_xtrain, new_xvalid, ytrain, yvalid, ans


def plot_stuff(BART, myNN, Bayes, KNN):
    """Plot comparison  between 4 methods in 5 instances

    Args:
        BART (float list): Accuracy of BART in 5 instances
        LSTM_NN (float list): Accuracy of LSTM_NN in 5 instances
        NaiveBayes (float list): Accuracy of NaiveBayes in 5 instances
        K-NearestNeighbors (float list): Accuracy of K-NN in 5 instances
    """

    # SET BAR & FIG SIZES
    barWidth = 0.2
    plt.subplots(figsize=(12, 6))

    # SET BAR POSITIONS ON X AXIS
    br1 = np.arange(len(BART))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # PLOT 4 METHODS
    plt.bar(br1, BART, color='r', width=barWidth,
            edgecolor='grey', label='BART')
    plt.bar(br2, myNN, color='g', width=barWidth,
            edgecolor='grey', label='LSTM Neural Net')
    plt.bar(br3, Bayes, color='b', width=barWidth,
            edgecolor='grey', label='Naive Bayes')
    plt.bar(br4, KNN, color='y', width=barWidth,
            edgecolor='grey', label='K-Nearest Neighbors')

    # DEFINE X AND Y AXIS
    plt.xlabel('Varying test sets', fontweight='bold', fontsize=15)
    plt.ylabel('Validation accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(BART))], ['', '', '', '', ''])
    ax = plt.gca()
    ax.set_ylim([0.6, 0.8])

    # PLOT
    plt.legend()
    plt.show()
