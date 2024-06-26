import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def remove_stop_words(input):
    """Remove unecessary words from English dictionary

    Args:
        input (string): One tweet

    Returns:
        string: Processed tweet
    """
    stop_words = set(stopwords.words('english'))

    words = input.split()
    filtered_words = [word for word in words if word not in stop_words]

    return ' '.join(filtered_words)


def remove_pattern(input):
    """Remove usernames starting with @

    Args:
        input (string): One tweet

    Returns:
        string: Processed tweet
    """
    return re.sub(r"@\S+", "", input)


def remove_URL(input):
    """Remove website adresses

    Args:
        input (string): One tweet

    Returns:
        string: Processed tweet
    """
    return re.sub(r"http\S+", "", input)


def remove_special(input):
    """Remove special caracters and numbers

    Args:
        input (string): One tweet

    Returns:
        string: Processed tweet
    """
    return re.sub(r"[^a-zA-Z#]", " ", input)


def remove_punctuation(input):
    """Remove hashtag, keep word

    Args:
        input (string): One tweet

    Returns:
        string: Processed tweet
    """
    return re.sub("#", "", input)


def remove_short(input):
    """Remove short words and lemmatize them

    Args:
        input (string): One tweet

    Returns:
        string: Processed tweet
    """
    stemmer = PorterStemmer()
    output = []

    for word in input.split():
        if len(word) > 1:
            word = stemmer.stem(word)
            output.append(word)

    return output


def cleanup(tweet):
    """Process a raw tweet into a clean and lemmatized array of words

    Args:
        input (string): One tweet

    Returns:
        string: Processed tweet
    """
    #tweet = remove_pattern(tweet)
    tweet = remove_URL(tweet)
    tweet = remove_special(tweet)
    tweet = remove_stop_words(tweet)
    tweet = remove_short(tweet)

    return tweet
