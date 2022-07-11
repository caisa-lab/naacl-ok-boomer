import pandas as pd
import csv
import numpy as np
from stanfordnlp.server import CoreNLPClient
from tqdm import tqdm
import re
import math

from classification.cross_validation_training import UserPredictionPipeline

NOT_DEFINED_LABEL = 'NOT_DEFINED'

class FrameAnnotator(object):

    def __init__(self, frame_path):
        self.d_frame = pd.read_pickle(frame_path, compression='infer')

    def save_frame(self, path):
        self.d_frame.to_pickle(path)

    def write_data(self, id_mapping, column_name):
        res_row = []
        for index, _ in self.d_frame.iterrows():
            if index in id_mapping.keys():
                res_row.append(id_mapping[index])
            else:
                raise Exception('Error in frame annotation')
        self.d_frame[column_name] = res_row


class DemoAnnotator(FrameAnnotator):

    def __init__(self, frame_path, label):
        super().__init__(frame_path)
        self.label = label

    def annotate(self):
        mapping = {}
        for index, _ in self.d_frame.iterrows():
            mapping[index] = 'test'
        self.write_data(mapping, self.label)

STANCE_MAPPING = {
    's_against': -2,
    'against': -1,
    #'stance_not_inferrable' : 0,
    'undecided': 0,
    'favor': 1,
    's_favor': 2,
}

class GroundTruthStanceAnnotator(FrameAnnotator):

    ANNOTATED_USERS_PATH = 'echo_chambers/annotated_users.csv'
    ANNOTATED_POSTS_PATH =  'echo_chambers/stance_annotated_p4.pkl'

    def __init__(self, frame_path, label, topics):
        super().__init__(frame_path)
        self.label = label
        self.topics = topics
        self.annotated_users = {}
        self.annotated_posts = {}
        self.load_users_labels()
        self.load_posts_labels()

    def load_users_labels(self):
        with open(GroundTruthStanceAnnotator.ANNOTATED_USERS_PATH, newline='') as csvfile:
            lines = csv.reader(csvfile, delimiter=';')
            for row in lines:
                if row[1] in self.annotated_users.keys():
                    self.annotated_users[row[1]].add(row[0])
                else:
                    self.annotated_users[row[1]] = {row[0]}

    def load_posts_labels(self):
        post_frame = pd.read_pickle(GroundTruthStanceAnnotator.ANNOTATED_POSTS_PATH, compression='infer')
        for index, row in post_frame.iterrows():
            if row['annotation'] == 'stance_not_inferrable':
                continue
            self.annotated_posts[index] = STANCE_MAPPING[row['annotation']]

    def annotate_user(self, post_tuples):
        found_stances = [self.annotated_posts[pt[0]] for pt in post_tuples if pt[0] in self.annotated_posts.keys()]
        return np.average(found_stances)

    def annotate(self):
        #Check if in annotated users
        res_map = {}
        for index, row in self.d_frame.iterrows():
            is_annotated = False
            for topic in self.topics:
                if topic in self.annotated_users.keys():
                    is_annotated = is_annotated or index in self.annotated_users[topic]
            if is_annotated:
                print(self.annotate_user([(doc[0], doc[4]) for doc in row['posts'] if doc[4] in self.topics]))
                res = self.annotate_user([(doc[0], doc[4]) for doc in row['posts'] if doc[4] in self.topics])
                if str(res) == 'nan':
                    res_map[index] = NOT_DEFINED_LABEL
                else:
                    res_map[index] = res
            else:
                res_map[index] = NOT_DEFINED_LABEL
        self.write_data(res_map, self.label)

class RegexLocationAnnotator(FrameAnnotator):

    MANUAL_LOCATION = []

    def __init__(self, frame_path, label):
        super().__init__(frame_path)
        self.label = label

    def location_from_parsed(self, parsed):
        for sentence in parsed.sentence:
            lemma_list = [ll.lemma.lower() for ll in sentence.token]
            for lid in range(len(lemma_list) - 3):
                loc_type = None
                if lemma_list[lid:lid + 3] == ['i', 'be', 'from']:
                    loc_type = 'i am from'
                elif lemma_list[lid:lid + 3] == ['i', 'live', 'in']:
                    loc_type = 'i live in'

                if loc_type is not None:
                    loc = []
                    pid = lid + 3
                    while pid < len(sentence.token) and (sentence.token[pid].ner == 'LOCATION' or \
                                                         sentence.token[pid].pos.startswith('NN') or \
                                                         sentence.token[pid].lemma == 'the' or \
                                                         sentence.token[pid].lemma in RegexLocationAnnotator.MANUAL_LOCATION):
                        loc.append(sentence.token[pid].word)
                        pid += 1
                    if loc == ['the'] or len(loc) == 0:
                        continue
                    # print(author + '\t' + subr + '\t' + ' '.join(loc))
                    return loc

    def annotate(self):
        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
                           timeout=300000, memory='8G') as client:
            res_map = {}
            for index, row in tqdm(self.d_frame.iterrows()):
                print(index)
                locations = set()
                for doc in row['posts']:
                    try:
                        doc_locs = self.location_from_parsed(client.annotate(doc[2]))
                        locations.update(doc_locs)
                    except Exception:
                        continue
                res_map[index] = locations
        self.write_data(res_map, self.label)


class RegexGenderAnnotator(FrameAnnotator):
    def __init__(self, frame_path, label):
        super().__init__(frame_path)
        self.label = label

    def gender_from_post(self, post):
        body = post.lower().replace('\n', ' ')
        if 'i\'m a guy' in body or \
            'i am a guy' in body or \
            'i am a male' in body or \
            'i\'m a male' in body or \
            'i am a man' in body or \
            'i\'m a man' in body or \
            'i am a boy' in body or \
            'i\'m a boy' in body or \
            'i am male' in body or \
            'i\'m male' in body:
            return 0
        if 'i\'m a girl' in body or \
            'i am a girl' in body or \
            'i\'m a gal' in body or \
            'i am a gal' in body or \
            'i am a female' in body or \
            'i\'m a female' in body or \
            'i am a woman' in body or \
            'i\'m a woman' in body or \
            'i am female' in body or \
            'i\'m female' in body:
            return 1
        return None

    def annotate(self):
        res_map = {}
        for index, row in tqdm(self.d_frame.iterrows()):
            found = False
            for doc in row['posts']:
                if '&gt' in doc[2] or 'AUTOMOD' in doc[2]:
                    continue
                try:
                    doc_gender = self.gender_from_post(doc[2])
                    if doc_gender is not None:
                        if found and not doc_gender == res_map[index]:
                            res_map[index] = NOT_DEFINED_LABEL
                        elif not found:
                            res_map[index] = doc_gender
                            found = True
                except Exception as e:
                    continue
            if index not in res_map.keys():
                res_map[index] = NOT_DEFINED_LABEL
        self.write_data(res_map, self.label)

class RegexAgeAnnotator(FrameAnnotator):
    def __init__(self, frame_path, label):
        super().__init__(frame_path)
        self.label = label

    def age_from_post(self, post):
        body = post.lower().replace('\n', ' ')
        match_o = re.match('.*?(i am|i\'m) (\\d+) (years|yrs|yr) old[^e].*?', body)
        if match_o:
             return int(match_o.groups()[1].strip())
        if 'i go to school' in body \
            or 'i am a teenager' in body \
            or 'i am a teen' in body \
            or 'i am in my twenties' in body \
            or 'i\'m a teenager' in body \
            or 'i\'m a teen' in body \
            or 'i\'m in my twenties' in body:
            return 20
        if 'i am in my thirties' in body \
            or 'i am in my 30s' in body\
            or 'i\'m in my thirties' in body \
            or 'i\'m in my 30s' in body:
            return 35
        if 'i am in my fifties' in body \
            or 'i am in my 50s' in body\
            or 'i\'m in my fifties' in body \
            or 'i\'m in my 50s' in body:
            return 55
        if 'i am in my sixties' in body \
            or 'i am in my 60s' in body\
            or 'i\'m in my sixties' in body \
            or 'i\'m in my 60s' in body:
            return 65
        if 'my grandson' in body \
            or 'my granddaughter' in body:
            return 60
        return None

    def label_fun(self, age):
        if age < 30:
            return 1.0
        elif age <= 45:
            return 2.0
        else:
            return 3.0

    def annotate(self):
        res_map = {}
        for index, row in tqdm(self.d_frame.iterrows()):
            found = False
            for doc in row['posts']:
                if '&gt' in doc[2] or 'AUTOMOD' in doc[2]:
                    continue
                try:
                    doc_age = self.label_fun(self.age_from_post(doc[2]))
                    if doc_age is not None:
                        if found and not res_map[index] == NOT_DEFINED_LABEL and not doc_age == res_map[index]:
                            res_map[index] = NOT_DEFINED_LABEL
                        elif not found:
                            res_map[index] = doc_age
                            found = True
                except Exception as e:
                    continue
            if index not in res_map.keys():
                res_map[index] = NOT_DEFINED_LABEL
        self.write_data(res_map, self.label)


class RegexReligionAnnotator(FrameAnnotator):
    def __init__(self, frame_path, label):
        super().__init__(frame_path)
        self.label = label

    def religion_from_post(self, post):
        body = post.lower().replace('\n', ' ')
        match_o = re.match('.*?(i am|i\'m) (a )?(christian|muslim|secular|atheist|agnostic|hindu|buddhist).*?', body)
        if match_o:
            return  match_o.groups()[2]
        return None

    def annotate(self):
        res_map = {}
        for index, row in tqdm(self.d_frame.iterrows()):
            for doc in row['posts']:
                try:
                    doc_rel = self.religion_from_post(doc[2])
                    if doc_rel is not None:
                        print(doc_rel)
                        res_map[index] = doc_rel
                except Exception as e:
                    continue
            if index not in res_map.keys():
                res_map[index] = NOT_DEFINED_LABEL
        self.write_data(res_map, self.label)

class PredictionAnnotator(FrameAnnotator):

    def __init__(self, frame_path, prediction_pipeline_path, label):
        super().__init__(frame_path)
        self.prediction_pipeline = PredictionAnnotator.init_prediciton_pipeline(prediction_pipeline_path)
        self.label = label

    @staticmethod
    def init_prediciton_pipeline(path):
        return UserPredictionPipeline.instantiate_from_pickle(path)

    def annotate(self):
        res_map = {}
        for index, row in tqdm(self.d_frame.iterrows()):
            res_map[index] = self.prediction_pipeline.predict_user([post[2] for post in row['posts']])
        self.write_data(res_map, self.label)
