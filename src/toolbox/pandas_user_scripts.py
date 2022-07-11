import pandas as pd
from .user_id_fetcher import DASPUserIdFetcher
from .user_post_fetcher import DASPUserPostFetcher
from .graph_metrics.social_graph import SocialGraphLoader

def init_dataset_from_db(platform, topics, min_posts, res_path, added_users=[]):
    fetcher = DASPUserIdFetcher(platform)
    user_id_set = set()
    for topic in topics:
        topic_user_ids = fetcher.get_topic_user_ids(topic, min_posts)
        user_id_set = set.union(user_id_set, topic_user_ids)
    print('Fetched ' + str(len(user_id_set)) + ' user ids!')
    user_id_set = set.union(user_id_set, set(added_users))
    user_ids = list(user_id_set)
    d_frame = pd.DataFrame({'user_id': user_ids}, index=user_ids)
    d_frame.to_pickle(res_path)

def fetch_posts_from_db(dataset_path, platform):
    d_frame = pd.read_pickle(dataset_path, compression='infer')
    uids = list(d_frame['user_id'])
    post_mapping = DASPUserPostFetcher(platform).fetch_user_posts(uids)
    new_col = []
    for uid in uids:
        new_col.append(post_mapping[uid])

    d_frame['posts'] = new_col
    d_frame.to_pickle(dataset_path)
