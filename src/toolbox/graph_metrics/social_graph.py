import networkx as nx
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import community as community_louvain
from networkx.algorithms.community import *
from toolbox.dasp_database_connection import DaspDatabaseConnection
from toolbox.dasp_database_connection import DatabaseAccessor
import datetime
import matplotlib.cm as cm
import copy

def chunk_data(to_chunk, chunk_size):
    return list((to_chunk[i:i + chunk_size] for i in range(0, len(to_chunk), chunk_size)))

class SocialGraphLoader(DatabaseAccessor):

    UID_CHUNK_SIZE = 100 # Defines how many users a fetched from the DB at the same time

    def __init__(self, origin):
        super().__init__(origin)
        self.user_interactions = {}

    def user_thread_posts_recursive(self, user_ids, topic_filter=None):
        if topic_filter is None:
            inner_select = "(SELECT public.posts.id FROM public.posts WHERE posts.author_id IN " + str(user_ids).replace(
                '[', '(').replace(']', ')') \
                + " AND posts.origin = \'" + self.origin + "\')"
        else:
            inner_select = "(SELECT public.posts.id FROM public.posts WHERE posts.author_id IN " + str(
                user_ids).replace('[', '(').replace(']', ')') \
                           + " AND posts.topic IN " + str(topic_filter).replace('[', '(').replace(']', ')') \
                           + " AND posts.origin = \'" + self.origin + "\')"

        db_q = "WITH RECURSIVE rel_posts AS ( " \
               "SELECT DISTINCT public.posts.id, " \
               "public.posts.parent_id, " \
               "public.posts.toplevel_id, " \
               "public.posts.author_id, " \
               "public.posts.timestamp, " \
               "public.posts.topic " \
               "FROM public.posts " \
               "WHERE posts.id IN " + inner_select + " " \
               "UNION ALL " \
               "SELECT DISTINCT p.id, " \
               "p.parent_id, " \
               "p.toplevel_id, " \
               "p.author_id, " \
               "p.timestamp, " \
               "p.topic " \
               "FROM public.posts p " \
               "JOIN rel_posts ON p.id = rel_posts.parent_id" \
               ") " \
               "SELECT DISTINCT * FROM rel_posts;"

        q_res = self.safe_db_queue(db_q)
        print(len(q_res))
        res_map = {}
        user_post_map = {}
        for uid in user_ids:
            user_post_map[uid] = []
        for post in q_res:
            res_map[post[0]] = post
            if post[3] in user_ids:
                user_post_map[post[3]].append(post[0])
        return res_map, user_post_map

    def get_interactions(self, uid, user_posts, thread_posts, write_to_file=False, res_data_path=None):
        interactions = []
        interactions_per_post = {}
        for user_post_id in user_posts:
            post = thread_posts[user_post_id]
            toplevel_id = post[2]
            traverse_post = post
            post_interactions = []
            while traverse_post[2] is not None:
                traverse_post = thread_posts[traverse_post[1]]
                interactions.append((traverse_post[3], traverse_post[0], traverse_post[4]))
                post_interactions.append((traverse_post[3], traverse_post[0], traverse_post[4]))
            interactions_per_post[user_post_id] = post_interactions
            # user_id,user_post_id,user_post_ts,interaction_id,interaction_post_id,interaction_post_timestamp,topic
            if toplevel_id is not None and not traverse_post[0] == toplevel_id:
                raise Exception('Sth went wrong!')
        if write_to_file:
            with open(os.path.join(res_data_path, uid + ".txt"), 'w') as res_file:
                for user_post_id, post_interactions in interactions_per_post.items():
                    for interaction in post_interactions:
                        res_file.write(
                            uid + ';' + user_post_id + ';' + str(thread_posts[user_post_id][4]) + ';' + interaction[
                            0] + ';' + interaction[1] + ';' + str(interaction[2]) + ';' + str(
                            thread_posts[user_post_id][5]) + '\n')
        return interactions

    def load_from_dasp_db(self, user_ids, topic_filter=None, write_to_file=False, res_file_path=None):
        self.init_db_connection()
        self.user_interactions = {}
        chunked_ids = chunk_data(user_ids, SocialGraphLoader.UID_CHUNK_SIZE)
        for uid_chunk in tqdm(chunked_ids):
            print("Processing user chunk...")
            print("Fetching thread posts...")
            thread_posts, user_posts = self.user_thread_posts_recursive(uid_chunk, topic_filter=topic_filter)
            print(len(thread_posts))
            for uid, post_ids in user_posts.items():
                res = self.get_interactions(uid, post_ids, thread_posts, write_to_file=write_to_file, res_data_path=res_file_path)
                self.user_interactions[uid] = res
        self.close_db_connection()

    @staticmethod
    def interaction_from_str(interaction_str):
        interaction_str = str(interaction_str)
        interaction_str = interaction_str.replace('\\n', '')
        interaction_str = interaction_str.replace('\'', '')
        data = interaction_str.strip().split(';')
        interacted_user = data[3]
        post_id = data[4]
        post_timestamp = datetime.datetime.strptime(data[2], '%Y-%m-%d %H:%M:%S')
        source_post_timestamp = datetime.datetime.strptime(data[5], '%Y-%m-%d %H:%M:%S')
        topic = data[6]
        return interacted_user, post_id, source_post_timestamp, topic

    def load_from_file(self, user_ids, social_interaction_folder, topic_filter=None):
        for uid in user_ids:
            with open(os.path.join(social_interaction_folder, uid + ".txt"), 'rb') as user_file:
                user_social_interactions = []
                for interaction_str in user_file.readlines():
                    interacted_user, post_id,  post_timestamp, topic = SocialGraphLoader.interaction_from_str(interaction_str)
                    if interacted_user in user_ids and (topic == topic_filter or topic_filter is None):
                        user_social_interactions.append((interacted_user, post_id, post_timestamp))
            self.user_interactions[uid] = user_social_interactions

    def get_graph(self):
        nodes = self.user_interactions.keys()
        edge_weight_map = {}
        # EDGES CREATED UNIQUELY???
        for uid, interactions in self.user_interactions.items():
            for interaction in interactions:
                interacted_user_id = interaction[0]
                if interacted_user_id in nodes:
                    if (uid, interacted_user_id) in edge_weight_map.keys():
                        edge_weight_map[(uid, interacted_user_id)] += 1
                    elif (interacted_user_id, uid) in edge_weight_map.keys():
                        edge_weight_map[(interacted_user_id, uid)] += 1
                    else:
                        edge_weight_map[(uid, interacted_user_id)] = 1

        return SocialGraph(nodes, edge_weight_map)


class SocialGraph(object):

    def __init__(self, nodes, edge_weight_map, partition=None):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        for edge, weight_val in edge_weight_map.items():
            if edge[0] == edge[1]:
                continue
            self.graph.add_edge(edge[0], edge[1], weight=weight_val)
        print((nx.info(self.graph)))
        self.partition = partition
        self.cmap = None
        self.metric_record = {}

    def largest_connected_component(self):
        """
        Extract the largest connected component in the networkX network.
        """
        user_set = max(nx.connected_components(self.graph), key=len)
        self.graph = self.graph.subgraph(user_set)

    def partition_meta_algorithm(self, profile, metric='modularity', max_communities=4, max_iters=10):
        best_partition = None
        best_score = -1
        for coms in range(4, max_communities+1):
            print('------------------')
            print('Trying: ' + str(coms))
            for i in range(max_iters):
                try:
                    self.partition_graph(profile, num_communities=coms)
                    score = np.average(list(self.score_partition(metric).values()))
                    print(score)
                    if score > best_score:
                        print('Highscore')
                        best_score = score
                        best_partition = copy.deepcopy(self.partition)
                except Exception as e:
                    print(e)
                    continue
        self.partition = best_partition

    def partition_graph(self, profile, num_communities=3):
        """
        Color the nodes of the network with a certain
        strategy/profile:
        Most of them are community detection algorithms.
        Available profiles:
            - "community_louvain"
            - "community_max_modularity"
            - "community_propagation"
            - "community_fluid"
            - "uniform"
        :param profile: The strategy to color the network nodes by
        """
        if profile == 'community_louvain':
            self.partition = community_louvain.best_partition(self.graph)
        elif profile == 'community_max_modularity':
            sets = modularity_max.greedy_modularity_communities(self.graph, weight='weight')
            mapping = {}

            for index, com_set in enumerate(sets):
                for uid in com_set:
                    mapping[uid] = index

            test_count = {}
            for user, cluster in mapping.items():
                if cluster in test_count.keys():
                    test_count[cluster] += 1
                else:
                    test_count[cluster] = 1
            logging.info([num for num in test_count.values() if num > 1])
            self.partition = mapping
        elif profile == 'community_propagation':
            sets = asyn_lpa_communities(self.graph, "weight")
            mapping = {}

            for index, com_set in enumerate(sets):
                for uid in com_set:
                    mapping[uid] = index

            test_count = {}
            for user, cluster in mapping.items():
                if cluster in test_count.keys():
                    test_count[cluster] += 1
                else:
                    test_count[cluster] = 1
            logging.info([num for num in test_count.values() if num > 1])
            self.partition = mapping
        elif profile == 'community_fluid':
            sets = asyn_fluidc(self.graph, num_communities)
            mapping = {}

            for index, com_set in enumerate(sets):
                for uid in com_set:
                    mapping[uid] = index

            self.partition = mapping
        elif profile == 'uniform':
            mapping = {}
            for uid in list(self.graph.nodes.keys()):
                mapping[uid] = 1
            self.partition = mapping
        else:
            raise Exception("Invalid profiling choice")

    def get_partition(self):
        """
        Returns the current network partition
        as a list of list of user ids.
        :return: The current network partition
        as a list of list of user ids.
        """
        collector = {}

        for user, assignment in self.partition.items():
            if assignment in collector.keys():
                collector[assignment].add(user)
            else:
                new_set = set()
                new_set.add(user)
                collector[assignment] = new_set

        return collector

    def color_nodes(self, color_palette='viridis'):
        if self.partition is None:
            raise Exception('No partition set for coloring.')

        print(max(self.partition.values()) + 1)
        self.cmap = cm.get_cmap(color_palette, (max(self.partition.values()) + 1))
        print(list(self.cmap.colors))

    def visualize(self):
        """
        Visualize the loaded network with pyplot.
        Can also be saved based on this.
        """
        if self.partition is None:
            raise Exception('No partition set for visualization.')

        if self.cmap is None:
            self.color_nodes()
        layout = nx.spring_layout(self.graph)
        plt.axis("off")
        nx.draw_networkx_nodes(self.graph, layout, self.partition.keys(), node_size=15,
                               cmap=self.cmap, node_color=list(self.partition.values()))
        nx.draw_networkx_edges(self.graph, layout, alpha=0.5)

    def record_metric(self, score_type, label):
        score = self.score_partition(score_type)
        self.metric_record[label] = score

    def record_user_degree(self):
        res_map = {}
        for node in self.partition.keys():
            res_map[node] = self.graph.degree(node)
        self.metric_record['node_degree'] = res_map

    def score_partition(self, score_type):
        """
        Score the given partition with the
        community scores separability and density.
        :param score_type:
        :return: The average separability, tuples of the intermediate scores (separability,density)
        """
        if score_type == 'modularity':
            community_collector = {}
            for node, community in self.partition.items():
                if community in community_collector.keys():
                    community_collector[community].append(node)
                else:
                    community_collector[community] = [node]
            modularity_val = nx.algorithms.community.quality.modularity(self.graph, community_collector.values(), weight=None)
            res = {}
            for com in community_collector.keys():
                res[com] = modularity_val
            return res
        if score_type == 'conductance':
            print(score_type)
            clean_edges = []
            for edge in self.graph.edges:
                if (edge[0], edge[1]) not in clean_edges and (edge[1], edge[0]) not in clean_edges:
                    clean_edges.append((edge[0], edge[1]))

            intra_edges = {}
            inter_edges = {}
            node_count = {}
            for node, community in self.partition.items():
                if community not in intra_edges.keys():
                    intra_edges[community] = 0
                    inter_edges[community] = 0

                if community in node_count.keys():
                    node_count[community] += 1
                else:
                    node_count[community] = 1

            for edge in clean_edges:
                if self.partition[edge[0]] == self.partition[edge[1]]:
                    intra_edges[self.partition[edge[0]]] += 1
                else:
                    inter_edges[self.partition[edge[0]]] += 1
                    inter_edges[self.partition[edge[1]]] += 1

            res_map = {}
            for community, inter in inter_edges.items():
                res_map[community] = inter_edges[community] / (2*intra_edges[community] + inter_edges[community])
            return res_map
        if score_type == 'expansion':
            print(score_type)
            clean_edges = []
            for edge in self.graph.edges:
                if (edge[0], edge[1]) not in clean_edges and (edge[1], edge[0]) not in clean_edges:
                    clean_edges.append((edge[0], edge[1]))

            intra_edges = {}
            inter_edges = {}
            node_count = {}
            for node, community in self.partition.items():
                if community not in intra_edges.keys():
                    intra_edges[community] = 0
                    inter_edges[community] = 0

                if community in node_count.keys():
                    node_count[community] += 1
                else:
                    node_count[community] = 1

            for edge in clean_edges:
                if self.partition[edge[0]] == self.partition[edge[1]]:
                    intra_edges[self.partition[edge[0]]] += 1
                else:
                    inter_edges[self.partition[edge[0]]] += 1
                    inter_edges[self.partition[edge[1]]] += 1

            res_map = {}
            for community, inter in inter_edges.items():
                res_map[community] = inter_edges[community] / node_count[community]
            return res_map
        if score_type == 'density':
            print(score_type)
            clean_edges = []
            for edge in self.graph.edges:
                if (edge[0], edge[1]) not in clean_edges and (edge[1], edge[0]) not in clean_edges:
                    clean_edges.append((edge[0], edge[1]))

            intra_edges = {}
            node_count = {}
            for node, community in self.partition.items():
                if community not in intra_edges.keys():
                    intra_edges[community] = 0

                if community in node_count.keys():
                    node_count[community] += 1
                else:
                    node_count[community] = 1

            for edge in clean_edges:
                if self.partition[edge[0]] == self.partition[edge[1]]:
                    intra_edges[self.partition[edge[0]]] += 1

            res_map = {}
            for community, intra in intra_edges.items():
                nodes = node_count[community]
                logging.info("Density:")
                logging.info(intra / (nodes * (nodes - 1) / 2))
                res_map[community] = intra / (nodes * (nodes - 1) / 2)
            return res_map
        if score_type == 'separability':
            print(score_type)
            clean_edges = []
            for edge in self.graph.edges:
                if (edge[0], edge[1]) not in clean_edges and (edge[1], edge[0]) not in clean_edges:
                    clean_edges.append((edge[0], edge[1]))

            intra_edges = {}
            inter_edges = {}
            node_count = {}
            for node, community in self.partition.items():
                if community not in intra_edges.keys():
                    intra_edges[community] = 0
                    inter_edges[community] = 0

                if community in node_count.keys():
                    node_count[community] += 1
                else:
                    node_count[community] = 1

            for edge in clean_edges:
                if self.partition[edge[0]] == self.partition[edge[1]]:
                    intra_edges[self.partition[edge[0]]] += 1
                else:
                    inter_edges[self.partition[edge[0]]] += 1
                    inter_edges[self.partition[edge[1]]] += 1

            res_map = {}
            for community, inter in inter_edges.items():
                logging.info("Separability:")
                logging.info(intra_edges[community]/inter)
                res_map[community] = intra_edges[community]/inter
            return res_map
        else:
            raise Exception('Invalid parameter!')

