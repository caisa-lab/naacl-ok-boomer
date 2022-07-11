from toolbox.dasp_database_connection import DatabaseAccessor

class DASPUserIdFetcher(DatabaseAccessor):

    def __init__(self, origin):
        super().__init__(origin)

    def get_active_user_ids(self, min_posts):
        self.init_db_connection()
        users_full = self.safe_db_queue("SELECT public.users.id, public.users.original_name , public.users.origin, COUNT(public.posts.id) " \
                   "FROM public.users " \
                   "INNER JOIN public.posts ON " \
                   "public.users.id = public.posts.author_id " \
                   "WHERE users.origin = '" + self.origin + "' " \
                                                       "GROUP BY public.users.id")
        self.close_db_connection()

        collector = []

        for user in users_full:
            if not user[0] == '[deleted]' and user[3] > min_posts:
                collector.append(user[0])
        return collector

    def get_topic_user_ids(self, filter_topic, min_posts):
        self.init_db_connection()
        users_full = self.safe_db_queue(
            "SELECT public.users.id, public.users.original_name , public.users.origin, COUNT(public.posts.id) " \
            "FROM public.users " \
            "INNER JOIN public.posts ON " \
            "public.users.id = public.posts.author_id " \
            "WHERE users.origin = '" + self.origin + "' " \
            "AND posts.topic = '"  + filter_topic + "' "
                                                     "GROUP BY public.users.id")
        self.close_db_connection()

        collector = []

        for user in users_full:
            if not user[0] == '[deleted]' and user[3] > min_posts:
                collector.append(user[0])
        return collector

