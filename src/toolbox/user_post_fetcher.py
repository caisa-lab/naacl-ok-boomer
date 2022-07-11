from toolbox.dasp_database_connection import DatabaseAccessor
from tqdm import tqdm

def chunk_data(to_chunk, chunk_size):
    return list((to_chunk[i:i + chunk_size] for i in range(0, len(to_chunk), chunk_size)))

class DASPUserPostFetcher(DatabaseAccessor):

    UID_CHUNK_SIZE = 400

    def __init__(self, origin):
        super().__init__(origin)

    def fetch_user_posts(self, uids):
        self.init_db_connection()
        res = {}
        uid_chunks = chunk_data(uids, DASPUserPostFetcher.UID_CHUNK_SIZE)
        for i in tqdm(range(len(uid_chunks))):
            uid_chunk = uid_chunks[i]
            db_q = "SELECT public.posts.id, public.posts.author_id, public.posts.content, public.posts.timestamp, public.posts.topic, public.posts.platform_specific " \
               "FROM public.posts " \
               "WHERE public.posts.origin = '" + self.origin + "' AND " \
               "public.posts.author_id IN " + str(uid_chunk).replace('[', '(').replace(']', ')')

            posts_full = self.safe_db_queue(db_q)

            for uid in uid_chunk:
                res[uid] = []

            for post in posts_full:
                res[post[1]].append(post)
        self.close_db_connection()
        return res

    def get_post_author_ids(self, pids):
        self.init_db_connection()
        pid_chunks = chunk_data(pids, DASPUserPostFetcher.UID_CHUNK_SIZE)
        res = set()
        for i in tqdm(range(len(pid_chunks))):
            pid_chunk = pid_chunks[i]
            db_q = "SELECT public.posts.id, public.posts.author_id, public.posts.topic " \
                   "FROM public.posts " \
                   "WHERE public.posts.origin = '" + self.origin + "' AND " \
                   "public.posts.id IN " + str(pid_chunk).replace('[', '(').replace(']', ')')
            q_res = self.safe_db_queue(db_q)
            for post in q_res:
                res.add((post[1], post[2]))
        self.close_db_connection()
        return res
