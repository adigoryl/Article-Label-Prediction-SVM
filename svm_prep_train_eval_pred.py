import requests
import nltk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from collections import defaultdict
from tqdm import tqdm

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score


def clean_n_split_data(data):
    """
    This method splits the dataset into two chunks, rows with full data (for training) and rows with missing data (for after-training prediction)
    Each row in data dict contains: id, title, content, publish_date, meta_description, sentiment, label
    :param data:
    :return:
    """
    full_data_rows = []
    missing_data_rows = []
    for i in range(len(data)):
        title = data[i]["title"]
        cont = data[i]["content"]
        meta_des = data[i]["meta_description"]
        sent = data[i]["sentiment"]
        label = data[i]["label"]

        if label == "no": label = 0
        if label == "yes": label = 1

        if None not in (title, cont, meta_des, sent, label) and "" not in (title, cont, meta_des, sent, label):

            full_data_rows.append({
                "title": title,
                "content": cont,
                "meta_des": meta_des,
                "sentiment": str(sent),
                "label": label
            })
        else:
            missing_data_rows.append({
                "title": title,
                "content": cont,
                "meta_des": meta_des,
                "sentiment": str(sent),
                "label": label
            })

    print("Rows without missing features: {}\nRows with missing features: {}".format(len(full_data_rows), len(missing_data_rows)))

    return full_data_rows, missing_data_rows


def prepare_for_svm(data):

    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['J'] = wordnet.ADJ
    tag_map['V'] = wordnet.VERB
    tag_map['R'] = wordnet.ADV

    wnl = WordNetLemmatizer()

    dataset = {
        "content": [],
        "label": []
    }

    for row in tqdm(data, desc="POS_tagging and Lemmization"):
        # We will be training the SVM based upon the content only
        article_str = word_tokenize(row["content"].lower())

        # We will be adding lemmanized and non-stopwords to this string.
        article_words = ""

        # Part-Of-Speech tag each word for lemmanization
        for word, tag in pos_tag(article_str):
            # Do not pass stop-words and non-alphabetic words
            if word not in stopwords.words('english') and word.isalpha():
                word = wnl.lemmatize(word, tag_map[tag[0]])
                article_words += str(word) + " "

        dataset["content"].append(article_words)
        dataset["label"].append(row["label"])

    return dataset


# Download data -> list[dict(id, title, content, publish_date, meta_description, sentiment, label)]
data = requests.get("https://jha-ds-test.s3-eu-west-1.amazonaws.com/legal/data.json").json()

# Data split to full rows and rows with missing labels
full_features_data, missing_features_data = clean_n_split_data(data)
print("Original dataset sample:\n{}".format(full_features_data[0]))

# Here we perform a word tokenization along with bunch of data cleaning techniques
full_data_rows = prepare_for_svm(full_features_data)
print("Cleaned and Lemmanized dataset sample:\n{}".format(full_data_rows["content"][0]))

# Split the full data rows into training and test ratio  0.8 : 0.2
train, test, train_labels, test_labels = model_selection.train_test_split(full_data_rows["content"],
                                                                          full_data_rows["label"],
                                                                          test_size=0.2)
# Transform corpus into -> Term Frequency - Inverse Term Frequency
vectorizer = TfidfVectorizer()
vectorizer.fit(full_data_rows["content"])

# Transform the training corpus into TfitFVec probabilities
train_freq = vectorizer.transform(train)
test_freq = vectorizer.transform(test)
print("Term Frequency - Inverse Term Frequency transformed input sample:\n{}".format(train_freq[0]))
print("{} Word tokens along with its index in the matrix map:\n{}".format(len(vectorizer.vocabulary_), vectorizer.vocabulary_))

# Training - Support Vector Machine
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_freq, train_labels)

# Test data predictions
predictions_SVM = SVM.predict(test_freq)

# Accuracy_score function to get the accuracy
print("SVM Accuracy Score: {}".format(accuracy_score(predictions_SVM, test_labels) * 100))
