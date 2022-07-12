from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np
import spacy
from spacymoji import Emoji
from nltk import tokenize
import logging
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from nltk import tokenize
import abc
from bert_serving.client import BertClient
import re

"""
This class bundles the different methodologies for feature vector calculation.
"""


class AbstractFeatureComputer(abc.ABC):
    """
    Abstract super class for the different types of feature computers.
    To be used in the abstract user prediction pipeline.
    """

    @abc.abstractmethod
    def transform(self, documents: [str]) -> []:
        """
        Calculate the feature vector of a user based on
        the list of the posts he produced,
        :param documents: The documents of the user to vectorize
        :return: The computed feature vector
        """
        pass


class TwitterTfIdfVectorizer(TfidfVectorizer):
    """
    Overrides the sklearn TfIdfVectorizer.
    Id adds a stemmer and a tweet tokenizer to the n_gram pipeline.
    """
    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.tweet_tokenizer = TweetTokenizer()
        self.token_stemmer = PorterStemmer()

    def stem(self, token: str):
        """
        Stem a given token using the PorterStemmer or the two hardcoded rules.

        :param token: Token to stem
        :return: Stemmed token
        """
        if token.startswith('@'):
            return "TWITTER_AT"
        if token.startswith('http'):
            return "LINK"
        else:
            return self.token_stemmer.stem(token)

    def build_tokenizer(self):
        """
        Overrides the build_tokenizer method.

        :return: The tokenizer lambda expression
        """
        return lambda doc: [self.stem(token) for token in self.tweet_tokenizer.tokenize(doc.strip('\"'))]


class BertAsServiceEmbedder(AbstractFeatureComputer):
    def __init__(self):
        self.client = BertClient()
        self.feature_dict = np.zeros(768)

    def doc_to_bert_interpertable(self, doc):
        """
        Helper method for transferring a given document
        into a bert-interpretable string, where senteces are
        separated by '|||'
        :param doc: The document to transform
        :return: The bert-interpretable string
        """
        sentences = tokenize.sent_tokenize(doc)
        res = ""
        for index, sentence in enumerate(sentences):
            res += sentence
            if index < len(sentences) - 1:
                res += ' ||| '
        return res

    def transform(self, documents: [str]) -> []:
        return np.average(self.client.encode([self.doc_to_bert_interpertable(doc) for doc in documents]), axis=0)


class BERTFeatureComputer(AbstractFeatureComputer):

    def __init__(self):
        self.model = SentenceTransformer('stsb-roberta-base')
        self.feature_dict = np.zeros(768)

    def transform(self, documents: [str]) -> []:
        return np.average(self.model.encode(documents), axis=0)


class TfIdfFeatureComputer(AbstractFeatureComputer):
    """
    Feature computer for the tf-idfs features.
    It utilises the sklearn tf-idf feature calculator
    """
    def __init__(self, train_data: []):
        """
        Constructor
        :param train_data: The documents that are used for training
        """
        self.feature_corpus = train_data
        self.feature_dict = []
        self.vectorizer = TwitterTfIdfVectorizer()
        self.build_feature_mapping(self.feature_corpus)
        logging.info(len(self.vectorizer.get_feature_names()))

    def build_feature_mapping(self, corpus: []):
        """
        Build a feature mapping based on the given corpus
        :param corpus: The documents to build the feaure mapping on
        """
        self.vectorizer.ngram_range = (1, 3)
        self.vectorizer.min_df = 80
        self.vectorizer.fit(corpus)
        self.feature_dict = [list() for i in range(len(self.vectorizer.vocabulary_))]

        for token, index in self.vectorizer.vocabulary_.items():
            self.feature_dict[index] = token

    def vectorize_data(self, data: []):
        """

        :param data:
        :return:
        """
        return [self.vectorizer.transform([document]).toarray()[0] for document in data]

    def transform(self, user_documents: []):
        """
        Calculate the average tf-idf feature vector of one user based on his/her given documents
        :param user_documents: The given documents of a user
        :return: The calculated average tf-idf feature vector
        """
        user_doc_vectors = self.vectorize_data(user_documents)
        user_doc_vectors = [np.array(vec) for vec in user_doc_vectors]
        avg = np.average(user_doc_vectors, axis=0)
        return avg

    def vectorize_bag_of_docs(self, data: [[]]):
        """
        Vectorize the data of multiple users
        :param data: The documents of the users as a list of lists
        :return: The calculatedfeature vectors in the same order
        """
        res_data = []
        for doc_set in data:
            user_doc_vectors = self.vectorize_data(doc_set)
            user_doc_vectors = [np.array(vec) for vec in user_doc_vectors]
            avg = np.average(user_doc_vectors, axis=0)
            res_data.append(avg)

        return res_data


class SurfaceFeatureComputer(AbstractFeatureComputer):
    """
    Feature computer for the abstract/surface features.
    Name those are:
    -The avg number of sentences
    -The avg number of emojis
    -The profanity ratio
    -The avg number of token
    -The avg number of ats
    -The avg number of links
    -The avg number of hashtags
    """

    def __init__(self):
        """
        Constructor
        """
        self.tweet_tokenizer = TweetTokenizer()
        self.token_stemmer = PorterStemmer()
        self.nlp = spacy.load('en_core_web_sm')
        emoji = Emoji(self.nlp)
        self.nlp.add_pipe(emoji, first=True)
        self.feature_dict = ["SENT", "EMOJI", "PROFANITY", "TOKEN", "AT", "LINK", "HASHTAG"]

    def transform(self, documents: []):
        """
        Calculate the average surface feature vector of one user based on his/her given documents
        :param user_documents: The given documents of a user
        :return: The calculated average surface feature vector
        """
        doc_tokens = [self.tokenize(doc) for doc in documents]

        vector = np.array(self.get_sent_based_averages(documents))
        vector = np.append(vector, [self.get_token_based_averages(doc_tokens)])

        return vector

    def tokenize(self, doc: str):
        """
        Helper method for tokenizing a given post.
        :param doc: The post to tokenize
        :return: The resulting tokens
        """
        return [self.token_stemmer.stem(token) for token in self.tweet_tokenizer.tokenize(doc.strip('\"'))]

    def get_token_based_averages(self, doc_tokens: [[]]):
        """
        Method for tracking all token based features.
        They are all tracked in the same loop for
        time efficiency reasons

        :param doc_tokens: The tokenized documents of a user
        :return: The token based feature vector
        """
        num_tokens_sum = 0
        num_ats_sum = 0
        num_hashtags_sum = 0
        num_links_sum = 0
        for tokenized in doc_tokens:
            num_tokens_sum = num_tokens_sum + len(tokenized)
            for token in tokenized:
                if token.strip().startswith('@'):
                    num_ats_sum = num_ats_sum + 1
                if token.strip().startswith('http'):
                    num_links_sum = num_links_sum + 1
                if token.strip().startswith('#'):
                    num_hashtags_sum = num_hashtags_sum + 1

        return [num_tokens_sum / len(doc_tokens),
                num_ats_sum / len(doc_tokens),
                num_hashtags_sum / len(doc_tokens),
                num_links_sum / len(doc_tokens)]

    def get_sent_based_averages(self, documents: []):
        """
        Method for tracking all sentence based features.
        They are all tracked in the same loop for
        time efficiency reasons

        :param documents: The posts of a user
        :return: The sentence based feature vector
        """
        sent_sum = 0
        emoji_sum = 0
        profanity_sum = 0
        politness_sum = 0
        for doc in documents:
            sent_sum = sent_sum + len(tokenize.sent_tokenize(doc))

            try:
                scanned = self.nlp(doc)
                emoji_sum = emoji_sum + len(scanned._.emoji)
            except ValueError:
                continue

            try:
                if profanity_check.predict([doc])[0] == 1:
                    profanity_sum = profanity_sum + 1
            except RuntimeError:
                continue

        return [sent_sum / len(documents),
                emoji_sum / len(documents),
                profanity_sum / len(documents)]


class WordToVecTopicVectorizer(AbstractFeatureComputer):
    """
    Feature computer for the word2vec-cluster features.
    """

    def __init__(self):
        """
        Constructor
        """
        self.tweet_tokenizer = TweetTokenizer()
        self.token_stemmer = PorterStemmer()
        self.word_to_vec_model = None
        self.cluster_mapping = {}
        self.cluster_amount = -1
        self.feature_dict = []

    def tokenize(self, doc: str):
        """
        Helper method for tokenizing a given document
        :param doc: The document to tokenize
        :return: The extracted tokens
        """
        return [self.token_stemmer.stem(token) for token in self.tweet_tokenizer.tokenize(doc.strip('\"'))]

    def tweet_to_tokenized_sentences(self, doc) -> [[]]:
        """
        Transfer a given post to a list of list with the tokens of each sentence
        in the post.

        :param doc: The post to tokenize
        :return: The split sentences as their a list of their tokens
        """
        res = []

        if not isinstance(doc, str):
            logging.warning("Unexpected type - skipping document")
            logging.warning(doc)
            return res

        for sent in tokenize.sent_tokenize(doc):
            sent_res = []
            sent_tokens = self.tokenize(sent)
            for i, token in enumerate(sent_tokens):
                if token.startswith('@'):
                    sent_tokens[i] = "TWITTER_AT"
                if token.startswith('http'):
                    sent_tokens[i] = "LINK"
            sent_res.extend(sent_tokens)
            res.append(sent_res)
        return res

    def fit(self, all_docs: []):
        """
        Fit the word2vec-cluster vectorizer.
        :param all_docs: All the documents for training
        """
        sentences = []

        for doc in all_docs:
            sentences.extend(self.tweet_to_tokenized_sentences(doc))

        self.word_to_vec_model = Word2Vec(sentences, min_count=1, sg=1)
        # Build word2vec dictionary
        w2v_indices = {word: self.word_to_vec_model.wv[word] for word in self.word_to_vec_model.wv.vocab}
        clustering_data = [*w2v_indices.values()]

        self.cluster_amount = 1000
        # Cluster word2vec dictionary
        kclusterer = MiniBatchKMeans(self.cluster_amount, max_iter=100, init_size=3000)
        logging.info("Clustering dictionary...")
        prediction_vector = kclusterer.fit_predict(clustering_data)

        index = 0
        for word, vec in w2v_indices.items():
            self.cluster_mapping[word] = prediction_vector[index]
            index += 1

        self.feature_dict = [list() for i in range(self.cluster_amount)]
        for word, cluster in self.cluster_mapping.items():
            self.feature_dict[cluster].append(word)

    def transform(self, user_documents: []):
        """
        Calculate the word2vec-cluster feature vector of one user based on his/her given documents
        :param user_documents: The given documents of a user
        :return: The calculated word2vec-cluster feature vector
        """
        res_vector = np.zeros(self.cluster_amount)

        sent_tokenized = [self.tweet_to_tokenized_sentences(doc) for doc in user_documents]
        token_count = 0

        for tokenized in sent_tokenized:
            for tokens in tokenized:
                token_count += len(tokens)
                for token in tokens:
                    if token in self.cluster_mapping.keys():
                        res_vector[self.cluster_mapping[token]] += 1

        if token_count == 0:
            return res_vector
        else:
            return res_vector / token_count


class BagOfWordsVectorizer(AbstractFeatureComputer):
    """
    Feature computer for the unigram bag of words features.
    """

    def __init__(self):
        """
        Constructor
        """
        self.word_mapping = {}
        self.tweet_tokenizer = TweetTokenizer()
        self.token_stemmer = PorterStemmer()
        self.feature_dict = []

    def tokenize(self, docs: []):
        """
        Helper method for tokenizing a given list of documents.
        :param docs: The list of documents to tokenize
        :return: The extracted tokens for each document
        """
        res = []
        for doc in docs:
            if not isinstance(doc, str):
                logging.warn("Unexpected type - skipping document")
                continue
            tokens = [self.token_stemmer.stem(token) for token in self.tweet_tokenizer.tokenize(doc.strip('\"'))]
            for index, token in enumerate(tokens):
                if token.startswith('@'):
                    tokens[index] = "TWITTER_AT"
                if token.startswith('http'):
                    tokens[index] = "LINK"
            res.append(tokens)
        return res

    def fit(self, documents_of_users: [[]]):
        """
        Fit the unigrams bag of words feautre computer
        to the given training data.
        :param documents_of_users: training data in the shape
        of a list of lists with each users documents
        """
        token_user_counter = {}

        index = 0
        for user_docs in documents_of_users:
            index += 1
            tokenized = self.tokenize(user_docs)

            for doc_tokens in tokenized:
                seen_tokens = []
                for token in doc_tokens:
                    if token not in seen_tokens:
                        seen_tokens.append(token)
                        if token in token_user_counter:
                            token_user_counter[token] = token_user_counter[token] + 1
                        else:
                            token_user_counter[token] = 1

        user_amount = len(documents_of_users)
        min_count = user_amount * 0.01

        logging.info("Min count:" + str(min_count))

        index = 0
        for token, count in token_user_counter.items():
            if count > min_count:
                self.word_mapping[token] = index
                index = index + 1

        self.feature_dict = [-1] * index
        for word, index in self.word_mapping.items():
            self.feature_dict[index] = word

    def transform(self, user_documents: []):
        """
        Calculate the unigrams feature vector of one user based on his/her given documents
        :param user_documents: The given documents of a user
        :return: The calculated unigrams feature vector
        """
        bow_vector = np.zeros(len(self.word_mapping.keys()))
        token_sum = 0
        for tokenized in self.tokenize(user_documents):
            token_sum += len(tokenized)
            for token in tokenized:
                if token in self.word_mapping.keys():
                    bow_vector[self.word_mapping[token]] = bow_vector[self.word_mapping[token]] + 1

        # Normalize
        bow_vector = bow_vector / len(user_documents)
        return bow_vector
