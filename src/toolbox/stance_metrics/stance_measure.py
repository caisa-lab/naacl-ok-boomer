import numpy as np
import os
from toolbox.dasp_database_connection import DaspDatabaseConnection
from toolbox.dasp_database_connection import DatabaseAccessor
import csv

default_topics = ["brexit",
                 "menrights",
                 "drugs",
                 "veganism-animalrights",
                 "healthcare",
                 "abortion",
                 "minimumwage",
                 "stemcell-cloning",
                 "adoption",
                 "capital-punishment",
                 "racism",
                 "climate-change",
                 "lgbtq",
                 "religion",
                 "police-brutality",
                 "presidential-race",
                 "nuclear-energy",
                 "guncontrol",
                 "palestine",
                 "feminism",
                 "fascism",
                 "capitalism",
                 "vaccines",
                 "migration",
                 "gentrification",
                 "freespeech"]

class StanceMeasure(DatabaseAccessor):

    def __init__(self, origin, topics=default_topics):
        super().__init__(origin)
        self.user_stances = {}
        self.topics = topics

    def fetch_user_posts(self, uid, post_filter=[]):
        #TODO: Already filter topics in queues
        if post_filter:
            select = "SELECT public.posts.id, public.posts.stance, public.posts.topic, public.posts.content, public.posts.timestamp FROM public.posts WHERE posts.author_id = \'" + uid + "\'" \
                + " AND posts. origin = \'" + self.origin + "\'" \
                + " AND posts.id in " +  str(post_filter).replace('[', '(').replace(']', ')')
        else:
            select = "SELECT public.posts.id, public.posts.stance, public.posts.topic, public.posts.content, public.posts.timestamp FROM public.posts WHERE posts.author_id = \'" + uid + "\'" \
                + " AND posts.origin = \'" + self.origin + "\'"

        self.init_db_connection()
        res = self.safe_db_queue(select)
        self.close_db_connection()
        return res

    def collect_stance_for_users_from_db(self, uids, post_filter=[], write_to_file=False, res_data_path=None):
        for uid in uids:
            print("Collecting user " + uid + "...")
            if not write_to_file:
                self.user_stances[uid] = self.create_stance_vector(self.fetch_user_posts(uid, post_filter=post_filter))
            else:
                res = self.fetch_user_posts(uid, post_filter=post_filter)
                with open(os.path.join(res_data_path, uid + ".txt"), "w") as target_file:
                    writer = csv.writer(target_file, delimiter=';', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                    for post_tuple in res:
                        writer.writerow(post_tuple)
                self.user_stances[uid] = self.create_stance_vector(res)

    def collect_stance_for_users_from_file(self, uids, data_folder, post_filter=[]):
        for uid in uids:
            print("Collecting user " + uid + "...")
            with open(os.path.join(data_folder, uid + ".txt"), "r") as target_file:
                reader = csv.reader(target_file, delimiter=';', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                posts = []
                for row in reader:
                    posts.append(row)
                if post_filter:
                    self.user_stances[uid] = self.create_stance_vector([row for row in posts if row[0] in post_filter])
                else:
                    self.user_stances[uid] = self.create_stance_vector(posts)

    def create_stance_vector(self, posts):
        posts_mapping = {}

        for post in posts:
            if not post[2] in self.topics:
                continue
            if post[2] in posts_mapping.keys():
                if post[1] is not None and post[1] is not "":
                    posts_mapping[post[2]].append(post[1])
            else:
                if post[1] is not None and post[1] is not "":
                    posts_mapping[post[2]] = [post[1]]
        res = {}

        for topic in self.topics:
            if topic in posts_mapping.keys():
                print(posts_mapping[topic])
                res[topic] = np.average(posts_mapping[topic])
            else:
                res[topic] = None

        return res

