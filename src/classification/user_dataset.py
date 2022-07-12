import pandas as pd
import numpy as np
from .feature_computing import SurfaceFeatureComputer
import logging

class UserDataset(object):
    """
    Encapsulation of an author profiling data set based
    on a pandas dataframe.
    """

    def __init__(self, dataframe, fold_amount=5):
        """
        Constructor.
        It automatically also folds the dataset for cross-validation.
        :param dataframe: The pandas dataframe that the dataset is based on
        :param fold_amount: The number of folds for the cross-validation
        """
        self.dataframe = dataframe
        self.chunked_ids = []
        self.num_folds = 0
        self.fold(fold_amount)

    def get_user(self, user_id: str):
        """
        Get a certain user from the dataset by his user_id.
        :param user_id: The user_id that is searched for
        :return:
        """
        try:
            return self.dataframe.loc[user_id]
        except:
            return {}

    def fold(self, amount=5) -> None:
        """
        Fold the dataset for cross-validation by the given number
        of folds.
        :param amount: The amount of folds for cross-validation
        """
        all_ids = self.dataframe['user_id'].tolist()
        self.chunked_ids = UserDataset.chunk_data(all_ids, amount)
        self.num_folds = amount

    def add_static_vector_column(self, header='static_vector'):
        """
        Adds a column 'static_vector' to the dataset that can be used
        for time efficient training of a machine learning model.
        This in this case is done for the surface/abstract features
        because they take the most time to be calculated while
        being the shortest vector.
        :param header: The header of the column
        """
        abstractvectorizer = SurfaceFeatureComputer()
        docs = [self.get_user(uid)['documents'] for uid in self.dataframe['user_id']]
        staticfeatures = [abstractvectorizer.transform(user_docs) for user_docs in docs]
        self.dataframe[header] = staticfeatures

    def add_agegroup_column(self, header='age_group', year_born=True):
        """
        Adds a column 'age_group' to the dataset based on a column
        'year_born' or 'age'.
        :param header: The header of the new column
        :param year_born: Whether to use year_born or age column as a source
        """
        curr_year = 2020
        if year_born:
            age_column = curr_year - np.array(self.dataframe['year_born'])
        else:
            age_column = np.array(self.dataframe['age'])
        age_group_column = []
        for user_age in age_column:
            if user_age <= 30:
                age_group_column.append(1.0)
                continue
            if user_age <= 45:
                age_group_column.append(2.0)
                continue
            else:
                age_group_column.append(3.0)

        self.dataframe[header] = age_group_column
        print(self.dataframe)

    def discretize_continous_value(self, key, borders, tag, average_binary=False):
        """
        Discretizes a column that is a continuous value into a certain
        amount of classes with the given borders.
        :param key: The header of the column to discretize
        :param borders: The borders of the the new classes
        :param tag: The header pf the new column
        :param average_binary: Only discretize into two classes, below and above average
        """
        cont_values = np.array(self.dataframe[key])
        res = []
        if average_binary:
            borders = [np.mean(cont_values)]
        for val in cont_values:
            added = False
            for index, border in enumerate(borders):
                if val < border:
                    res.append(index)
                    added = True
                    break
            if not added:
                res.append(len(borders))

        self.dataframe[tag] = res
        print(self.dataframe)

    def filter_column_values(self, column, value_to_filter, mode='eq'):
        """
        Remove users from the datas set with some criteria.
        :param column: The column based on which to filter
        :param value_to_filter: The threshold value to filter with
        :param mode: The filtering mode
        - 'eq': Remove if equal to value_to_filter
        - 'lt': Remove if less than value_to_filter
        - 'gt': Remove if greater than value_to_filter
        """
        values = np.array(self.dataframe[column])
        ids = np.array(self.dataframe['user_id'])
        remove = []
        for index, val in enumerate(values):
            if mode == 'eq':
                if val == value_to_filter:
                    remove.append(ids[index])
            elif mode == 'lt':
                if val < value_to_filter:
                    remove.append(ids[index])
            elif mode == 'gt':
                if val > value_to_filter:
                    remove.append(ids[index])

        for ind in remove:
            self.dataframe.drop(ind, inplace=True)

        print(self.dataframe)

    def get_fold(self, fold_index: int):
        """
        Get the old of index fold_index.
        :param fold_index: The fold index to retrieve
        :return: training_ids, testing_ids
        """
        return UserDataset.fold_data(self.chunked_ids, fold_index)

    def clean(self, remove_no_docs=True, remove_nan_label=True, remove_zero_label=False):
        """
        Clean the given dataset with some constraints.
        :param remove_no_docs: Remove user if he/she has no documents.
        :param remove_nan_label: Remove user if he/she has a nan label
        :param remove_zero_label: Remove user if he/she has a zero label
        """
        to_remove = []
        for uid in self.dataframe['user_id']:
            user = self.get_user(uid)
            if remove_no_docs and not user['documents']:
                to_remove.append(uid)
                continue
            if remove_nan_label:
                for label in self.dataframe.columns:
                    if str(user[label]).lower().strip() == "nan":
                        to_remove.append(uid)
            if remove_zero_label:
                for label in self.dataframe.columns:
                    try:
                        if float(user[label]) == 0:
                            to_remove.append(uid)
                    except:
                        continue

        logging.info("Cleaning dataset; remove count = " + str(len(to_remove)))

        for uid in to_remove:
            try:
                self.dataframe.drop(uid, inplace=True)
            except KeyError:
                continue

        print(self.dataframe)

    def print_column_distribution(self, column_name: str):
        """
        Print the distribution of a given column.
        :param column_name: Header of the column to print the distribution of
        """
        logging.info("Distribution for " + column_name)
        vals = self.dataframe[column_name]
        mapping = {}
        for val in vals:
            if val in mapping.keys():
                mapping[val] += 1
            else:
                mapping[val] = 1

        for val, num in mapping.items():
            logging.info(str(val) + ": " + str(num/len(vals)))

    def store_to_pickle(self, path: str):
        """
        Store the datas set to a pickle.
        :param path: The path to store the data set at
        """
        self.dataframe.to_pickle(path)

    @staticmethod
    def instantiate_from_pickle(path: str, fold_amount=5):
        """
        Instantiate the data set from given pickle path.
        :param path: Path to load the data set from
        :param fold_amount: Amount of folds for cross-validation
        :return: The instantiated data set
        """
        frame = pd.read_pickle(path)
        print(frame)
        return UserDataset(frame, fold_amount)

    @staticmethod
    def instantiate_from_dataset_files(user_label_file, user_document_file):
        """
        Load the data set for the first time from the
        two discribing files.
        This should only be used once, afterwards it should be saved
        and loaded as a pickle.
        :param user_label_file: File where each user id is annotated for all the header
        :param user_document_file: file that contains all the documents with their authors
        :return: The instantiated data set
        """
        raise NotImplementedError
        

    @staticmethod
    def fold_data(chunked_data: [[]], fold_index: int):
        """
        Fold the data into two list, one for training
        and one for testing during cross-validation.
        the fold_index determines the index of the cunk that is used as the
        testing set.
        :param chunked_data: The chunked data.
        :param fold_index: The index of the current fold
        :return: training_ids, testing_ids
        """
        train_data = []
        test_data = []

        for index in range(0, len(chunked_data)):
            if index == fold_index:
                test_data.extend(chunked_data[index])
            else:
                train_data.extend(chunked_data[index])

        return train_data, test_data

    def merge_classes(self, key, new_key, new_dist):
        """
        Merge together classes of a discretely distributed
        column of the dataset.
        :param key: The key of the column to merge the classes in
        :param new_key: The key of the column with the merged classes
        :param new_dist: The new merged classes. If the classes 1 and 2 should be merged it
        would for instance look like this: [[1, 2], [3]]
        """
        dist_map = {}
        for index, label_list in enumerate(new_dist):
            for label in label_list:
                dist_map[label] = index
        logging.info(dist_map)
        old_vals = self.dataframe[key]
        new_vals = [dist_map[val] for val in old_vals]
        self.dataframe[new_key] = new_vals

    def exclude_topic_posts(self):
        """
        Method for mitigating the topic bias in
        the author profiling. This is achieved by removing all
        the posts from the users that are predicte to be about
        one of the controversial topics.
        Note: Very time consuming
        """
        model_path = "data/reddit_model_simple_300.pt"
        threshold = 8
        classifier = SimpleTextClassifier.restore_from_file(model_path)
        user_posts = self.dataframe['documents']
        filtered_docs = []
        for docs in user_posts:
            try:
                filtered = [doc for doc in docs if classifier.predict_with_threshold(doc, threshold) is None]
            except:
                filtered_docs.append(docs)
                continue
            if not filtered:
                # Quick fix for error prevention and too not lose too many users
                filtered.append(docs[0])
            filtered_docs.append(filtered)

        self.dataframe['documents'] = filtered_docs
        print(self.dataframe)

    @staticmethod
    def join_datasets(dataframes, fold_amount=5):
        """
        Join together multiple data sets.
        The resulting data set only contains
        headers that are present in ALL the merged
        dataframes.
        :dataframes: The list of pandas dataframes to join together
        :return: The merged dataframes as a new folded UserDataset
        """
        headers = list(dataframes[0].columns.values)
        for frame in dataframes:
            headers_new = []
            for header in frame.columns.values:
                if header in headers:
                    headers_new.append(header)
            headers = headers_new

        logging.info("Merging dataframes on: " + str(headers) + "...")
        joined_frame = pd.concat(dataframes)
        joined_frame = joined_frame[headers]
        joined_frame = joined_frame.drop_duplicates('user_id')
        joined_frame = joined_frame.sample(frac=1)
        joined = UserDataset(joined_frame, fold_amount)
        print(joined.dataframe)
        return joined

    @staticmethod
    def chunk_data(to_chunk, amount):
        """
        Static helper method for chunking an array into a given amount
        of equally sized chunks.
        :param to_chunk: The array to chunk
        :param amount: The amount of chunks
        :return: The chunked array as a list of lists
        """
        chunk_size = len(to_chunk) // amount
        chunk_size = max(1, chunk_size)
        return list((to_chunk[i:i + chunk_size] for i in range(0, len(to_chunk), chunk_size)))
