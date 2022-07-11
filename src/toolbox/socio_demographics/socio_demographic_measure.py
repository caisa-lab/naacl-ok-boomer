from classification.cross_validation_training import UserPredictionPipeline
from toolbox.dasp_database_connection import DatabaseAccessor
import os
import csv

class SocioDemographicMeasure(DatabaseAccessor):

    classifiers = {
        'gender': UserPredictionPipeline.instantiate_from_pickle('data/model/socio-demographics/gender_no_topic.pkl')
    }

    def __init__(self, origin):
        super().__init__(origin)
        self.user_mapping = {}

    def get_user_posts(self, uid):
        select = "SELECT public.posts.id, public.posts.content FROM public.posts WHERE posts.author_id = \'" + uid + "\'" \
                 + " AND posts.origin = \'" + self.origin + "\')"

        return self.safe_db_queue(select)

    @staticmethod
    def predict_socio_demographics(user_posts, content_index=1):
        res = {}
        for dim, classifier in SocioDemographicMeasure.classifiers.items():
            if user_posts:
                res[dim] = classifier.predict_user([post[content_index] for post in user_posts])
            else:
                res[dim] = None
        return res

    def collect_user_socio_demographics_from_db(self, uids):
        for uid in uids:
            self.user_mapping[uid] = self.predict_socio_demographics(self.get_user_posts(uid))

    def collect_user_socio_demographics_from_file(self, uids, data_folder):
        for uid in uids:
            print("Classifying user " + uid + "...")
            with open(os.path.join(data_folder, uid + ".txt"), "r") as target_file:
                reader = csv.reader(target_file, delimiter=';', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                posts = []
                for row in reader:
                    posts.append(row)
                self.user_mapping[uid] = self.predict_socio_demographics(posts, content_index=3)
