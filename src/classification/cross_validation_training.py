import random
import pickle
import numpy as np
import logging
from .user_dataset import UserDataset
from .feature_computing import TfIdfFeatureComputer
from .feature_computing import BagOfWordsVectorizer
from .feature_computing import WordToVecTopicVectorizer
from .feature_computing import AbstractFeatureComputer


class UserPredictionPipeline(object):
    """
    Class to encapsulate a complete author profiling pipeline, to be able
    to save, instantiate and use the trained classifiers with the
    correct feature vector approached
    """

    def __init__(self, model, feature_computer: AbstractFeatureComputer, feauture_computer_type, label: str):
        """
        Constructor
        :param model: The nltk model to execute in the pipeline
        :param feature_computer: The feature computer to use on the input documents of a user
        :param feauture_computer_type: Helper-variable indicating which feature vector approach is used
        :param label: The predicted type of label
        """
        self.model = model
        self.feature_computer_type = feauture_computer_type
        self.feature_computer = feature_computer
        self.label = label

    def predict_user(self, user_documents: [str], min_proba=None):
        """
        Execute the encapsulated author profiling pipeline on the documents of a given user
        :param user_documents: The documents of the user to profile
        :param min_proba: The minimum probability array
                        Format: [min_confidence, array_index of predict_proba vector, fallback predicted class]
        :return: The predicted class
        """
        feature_vector = self.feature_computer.transform(user_documents)
        predicted_class = self.model.predict([feature_vector])[0]
        if min_proba is not None:
            proba = self.model.predict_proba([feature_vector])[0]
            if proba[min_proba[1]] < min_proba[0]:
                return min_proba[2]
        return predicted_class

    def store_to_pickle(self, path: str):
        """
        Store the pipeline to a give path
        :param path: The path to store the pipeline at
        """
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def instantiate_from_pickle(path: str):
        """
        Instantiate a ready-to-run pipeline from a given patj
        :param path: Path to instantiate the pipeline from
        :return: The instantiate author profiling pipeline
        """
        loaded_instance = pickle.load(open(path, "rb"))
        print(type(loaded_instance.model))
        return loaded_instance

def print_priors(label_vec: []):
    """
    Helper-method for printing the prior class distribution
    of a given label vector
    :param label_vec:
    """
    label_counter = {}

    for label in label_vec:
        if label in label_counter.keys():
            label_counter[label] = label_counter[label] + 1
        else:
            label_counter[label] = 1

    for label, amount in label_counter.items():
        res = (amount / len(label_vec))
        logging.info(str(label) + ": " + str(res))


def flatten(input: [[]]):
    """
    Helper method to "flatten" a list of lists
    """
    res_data = []
    for list in input:
        res_data.extend(list)
    return res_data


def balance_dataset_down(training_vectors: [], training_labels: []):
    """
    Helper-method to balance down a given data set of vectors and labels.
    :param training_vectors: The vectors of the given data set
    :param training_labels: The labels of the given data set
    :return: The balanced down data set
    """
    label_map = {}
    dataset = []
    for index in range(0, len(training_labels)):
        dataset.append((training_vectors[index], training_labels[index]))

    for data_tuple in dataset:
        if data_tuple[1] == '':
            continue
        elif data_tuple[1] in label_map:
            label_map[data_tuple[1]].append(data_tuple)
        else:
            label_map[data_tuple[1]] = [data_tuple]

    min_len = min([len(docs) for docs in label_map.values()])

    balanced_dataset = []

    for docs in label_map.values():
        random.shuffle(docs)
        balanced_dataset.extend(docs[:min_len])

    random.shuffle(balanced_dataset)
    return [elem[0] for elem in balanced_dataset], [elem[1] for elem in balanced_dataset]


def balance_dataset_up(training_vectors: [], training_labels: []):
    """
    Helper-method to balance down a given data set of vectors and labels.
    It uses random sampling for the balancing.
    :param training_vectors: The vectors of the given data set
    :param training_labels: The labels of the given data set
    :return: The balanced up data set
    """
    label_map = {}
    dataset = []
    for index in range(0, len(training_labels)):
        dataset.append((training_vectors[index], training_labels[index]))

    for data_tuple in dataset:
        if data_tuple[1] == '':
            continue
        elif data_tuple[1] in label_map:
            label_map[data_tuple[1]].append(data_tuple)
        else:
            label_map[data_tuple[1]] = [data_tuple]

    max_len = max([len(docs) for docs in label_map.values()])
    balanced_dataset = []

    for docs in label_map.values():
        added = len(docs)
        balanced_dataset.extend(docs)
        while added < max_len:
            added = added + 1
            balanced_dataset.append(random.choice(docs))

    random.shuffle(balanced_dataset)
    return [elem[0] for elem in balanced_dataset], [elem[1] for elem in balanced_dataset]


def domain_adaption_feature(embedding, is_source):
    if is_source:
        return np.concatenate([embedding, embedding, np.zeros(len(embedding))])
    else:
        return np.concatenate([embedding, np.zeros(len(embedding)), embedding])


class CrossValidationTrainer(object):
    """
    The training environment for the specific task of author profiling.
    It enables training of multiple models on different feature vector
    calculation approaches using cross-validation.
    See: user_dataset.py
    """

    def __init__(self, training_user_dataset: UserDataset, models: [], label: str):
        """
        Constructor
        :param training_user_dataset: The UserDataset that should be trained on
        :param models: The model that should be trained
        :param label: The column of the user data set to use a the label
        """
        self.dataset = training_user_dataset
        self.models = models
        self.label = label
        self.vectorizer = None

    def domain_adaption_cross_validation_training(self, target_domain_dataset, fold_amount: int,
                                                  feature_computer='unigrams'):
        self.dataset.fold(amount=fold_amount)
        # target_domain_dataset.fold(amount=fold_amount)

        for fold_index in range(0, self.dataset.num_folds):
            logging.info("Fold: " + str(fold_index))

            training_ids, testing_ids = self.dataset.get_fold(fold_index)
            target_training_ids, target_testing_ids = target_domain_dataset.get_fold(fold_index)

            if feature_computer == 'static-vector':
                training_features = [
                    domain_adaption_feature(self.dataset.get_user(uid)['static_vector'], is_source=True) for uid in
                    training_ids]
                testing_features = [domain_adaption_feature(self.dataset.get_user(uid)['static_vector'], is_source=True)
                                    for uid in testing_ids]

            elif feature_computer == 'unigrams':
                self.vectorizer = BagOfWordsVectorizer()
                joined_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                joined_docs.extend([target_domain_dataset.get_user(uid)['documents'] for uid in target_training_ids])
                self.vectorizer.fit(joined_docs)

                training_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                training_features = [domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=True) for
                                     user_docs in training_docs]

                testing_docs = [self.dataset.get_user(uid)['documents'] for uid in testing_ids]
                testing_features = [domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=True) for
                                    user_docs in testing_docs]

                target_training_docs = [target_domain_dataset.get_user(uid)['documents'] for uid in target_training_ids]
                target_training_features = [
                    domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=False) for
                    user_docs in target_training_docs]

                target_testing_docs = [target_domain_dataset.get_user(uid)['documents'] for uid in target_testing_ids]
                target_testing_features = [
                    domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=False) for
                    user_docs in target_testing_docs]

                training_features.extend(target_training_features)
                testing_features.extend(target_testing_features)

            elif feature_computer == 'avg-tf-idf':
                fit_data = flatten([self.dataset.get_user(uid)['documents'] for uid in training_ids])
                self.vectorizer = TfIdfFeatureComputer(fit_data)

                training_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                training_features = domain_adaption_feature(self.vectorizer.vectorize_bag_of_docs(training_docs),
                                                            is_source=True)

                testing_docs = [self.dataset.get_user(uid)['documents'] for uid in testing_ids]
                testing_features = domain_adaption_feature(self.vectorizer.vectorize_bag_of_docs(testing_docs),
                                                           is_source=True)

            elif feature_computer == 'word2vec-cluster':
                self.vectorizer = WordToVecTopicVectorizer()
                joined_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                joined_docs.extend([target_domain_dataset.get_user(uid)['documents'] for uid in target_training_ids])
                self.vectorizer.fit(flatten(joined_docs))

                training_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                training_features = [domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=True) for
                                     user_docs in training_docs]

                testing_docs = [self.dataset.get_user(uid)['documents'] for uid in testing_ids]
                testing_features = [domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=True) for
                                    user_docs in testing_docs]

                target_training_docs = [target_domain_dataset.get_user(uid)['documents'] for uid in target_training_ids]
                target_training_features = [
                    domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=False) for
                    user_docs in target_training_docs]

                target_testing_docs = [target_domain_dataset.get_user(uid)['documents'] for uid in target_testing_ids]
                target_testing_features = [
                    domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=False) for
                    user_docs in target_testing_docs]

                training_features.extend(target_training_features)
                testing_features.extend(target_testing_features)

            elif feature_computer == 'prebuilt':
                training_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                training_features = [domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=True) for
                                     user_docs in training_docs]

                testing_docs = [self.dataset.get_user(uid)['documents'] for uid in testing_ids]
                testing_features = [domain_adaption_feature(self.vectorizer.transform(user_docs), is_source=True) for
                                    user_docs in testing_docs]
            else:
                raise ValueError(feature_computer + " is not a valid argument")

            training_labels = [self.dataset.get_user(uid)[self.label] for uid in training_ids]
            training_labels.extend([target_domain_dataset.get_user(uid)[self.label] for uid in target_training_ids])
            training_features, training_labels = balance_dataset_down(training_features, training_labels)
            print_priors(training_labels)

            logging.info("training_user_size: " + str(len(training_features)))

            testing_labels = [self.dataset.get_user(uid)[self.label] for uid in testing_ids]
            testing_labels.extend([target_domain_dataset.get_user(uid)[self.label] for uid in target_testing_ids])
            testing_features, testing_labels = balance_dataset_down(testing_features, testing_labels)

            logging.info("testing_user_size: " + str(len(testing_features)))

            [model.fit(training_features, training_labels) for model in self.models]

            for model in self.models:
                fold_score = model.score(testing_features,
                                         testing_labels)
                logging.info("Accuracy: " + str(fold_score))

    def cross_validation_training(self, fold_amount: int, feature_computer='unigrams'):
        """
        Trains the given models on the given user data set performing fold_amount
        of cross validation steps. Accuracies of the models are printed for
        each iteration.
        :param fold_amount: The amount of cross-validation iterations to do
        :param feature_computer: The type of feature computer to user
        Available types:
            'static-vector': Use what is save under 'static_vector'
                             in the user data set that is used for training
            'unigrams': See feature_computing.py
            'avg-tf-idf': See feature_computing.py
            'word2vec-cluster': See feature_computing.py
            'prebuilt': Use the vectorizer that was priorly set as the vectorizer
                        for this instance of the class as the feature computer
        """
        self.dataset.fold(amount=fold_amount)

        for fold_index in range(0, self.dataset.num_folds):
            logging.info("Fold: " + str(fold_index))

            training_ids, testing_ids = self.dataset.get_fold(fold_index)

            if feature_computer == 'static-vector':
                training_features = [self.dataset.get_user(uid)['static_vector'] for uid in training_ids]
                testing_features = [self.dataset.get_user(uid)['static_vector'] for uid in testing_ids]

            elif feature_computer == 'unigrams':
                self.vectorizer = BagOfWordsVectorizer()
                self.vectorizer.fit([self.dataset.get_user(uid)['documents'] for uid in training_ids])

                training_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                training_features = [self.vectorizer.transform(user_docs) for user_docs in training_docs]

                testing_docs = [self.dataset.get_user(uid)['documents'] for uid in testing_ids]
                testing_features = [self.vectorizer.transform(user_docs) for user_docs in testing_docs]

            elif feature_computer == 'avg-tf-idf':
                fit_data = flatten([self.dataset.get_user(uid)['documents'] for uid in training_ids])
                self.vectorizer = TfIdfFeatureComputer(fit_data)

                training_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                training_features = self.vectorizer.vectorize_bag_of_docs(training_docs)

                testing_docs = [self.dataset.get_user(uid)['documents'] for uid in testing_ids]
                testing_features = self.vectorizer.vectorize_bag_of_docs(testing_docs)

            elif feature_computer == 'word2vec-cluster':
                self.vectorizer = WordToVecTopicVectorizer()
                self.vectorizer.fit(flatten([self.dataset.get_user(uid)['documents'] for uid in training_ids]))

                training_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                training_features = [self.vectorizer.transform(user_docs) for user_docs in training_docs]

                testing_docs = [self.dataset.get_user(uid)['documents'] for uid in testing_ids]
                testing_features = [self.vectorizer.transform(user_docs) for user_docs in testing_docs]

            elif feature_computer == 'prebuilt':
                training_docs = [self.dataset.get_user(uid)['documents'] for uid in training_ids]
                training_features = [self.vectorizer.transform(user_docs) for user_docs in training_docs]

                testing_docs = [self.dataset.get_user(uid)['documents'] for uid in testing_ids]
                testing_features = [self.vectorizer.transform(user_docs) for user_docs in testing_docs]
            else:
                raise ValueError(feature_computer + " is not a valid argument")

            training_labels = [self.dataset.get_user(uid)[self.label] for uid in training_ids]
            training_features, training_labels = balance_dataset_down(training_features, training_labels)
            print_priors(training_labels)

            logging.info("training_user_size: " + str(len(training_features)))

            testing_labels = [self.dataset.get_user(uid)[self.label] for uid in testing_ids]
            testing_features, testing_labels = balance_dataset_down(testing_features, testing_labels)

            logging.info("testing_user_size: " + str(len(testing_features)))

            [model.fit(training_features, training_labels) for model in self.models]

            for model in self.models:
                fold_score = model.score(testing_features,
                                         testing_labels)
                logging.info("Accuracy: " + str(fold_score))


def evaluate_model_coefficients(model_type: str, model_coefficients: [], feature_mapping: [], top_n=10):
    """
    Method that bundles the algorithms to evaluate the feature weights of different
    machine learning models

    Available model types:
        'svm'
        'forest'
    :param model_type: Model type to evaluate the weights of
    :param model_coefficients: The model coefficients to evaluate
    :param feature_mapping: The features (e.g. unigrams) that belong the the weight of the same index
    :param top_n: The amount of 'top' features to print out
    """
    if model_type == 'svm':
        mapped = []

        for index, coefficient in enumerate(model_coefficients):
            mapped.append((coefficient, feature_mapping[index]))

        mapped.sort(key=lambda tup: tup[0])

        print("Bottom " + str(top_n) + ":")
        for entry in mapped[:top_n]:
            print(str(round(entry[0], 3)) + " & " + str(entry[1]) + " \\\\")
        print("Top " + str(top_n) + ":")
        for entry in reversed(mapped[-top_n:]):
            print(str(round(entry[0], 3)) + " & " + str(entry[1]) + " \\\\")
    elif model_type == 'forest':
        importances = model_coefficients
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(top_n):
            print("%d. feature %s (%f)" % (f + 1, feature_mapping[indices[f]], importances[indices[f]]))
    else:
        raise Exception(model_type + " is not a valid model_type")
