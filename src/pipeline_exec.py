import os
from toolbox.pandas_user_scripts import init_dataset_from_db
from toolbox.pandas_user_scripts import fetch_posts_from_db
from toolbox.graph_metrics.social_graph import SocialGraphLoader
from toolbox.frame_annotator import GroundTruthStanceAnnotator
from toolbox.frame_annotator import RegexLocationAnnotator
from toolbox.frame_annotator import *
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pickle as pkl
import matplotlib.cm as cm
import scipy

from topic_result_writer import TopicResultWriter

annotated_user_map = []

def load_annotated_users():
    with open('echo_chambers/annotated_users.csv', newline='') as csvfile:
        lines = csv.reader(csvfile, delimiter=';')
        for row in lines:
            annotated_user_map.append((row[0], row[1]))

load_annotated_users()

PLATFORM = 'REDDIT'

SOURCE_PATH = 'echo_chambers/'

TOPICS = [
    (['abortion'], 'echo_chambers/abortion/min_10.pkl', 'echo_chambers/abortion/graph'),
    (['climate-change'], 'echo_chambers/climate-change/min_10.pkl', 'echo_chambers/climate-change/graph'),
    (['feminism', 'menrights'], 'echo_chambers/feminism/min_10.pkl', 'echo_chambers/feminism/graph'),
    (['guncontrol'], 'echo_chambers/guncontrol/min_10.pkl', 'echo_chambers/guncontrol/graph'),
    (['veganism-animalrights'], 'echo_chambers/veganism-animalrights/min_10.pkl', 'echo_chambers/veganism-animalrights/graph'),
    (['brexit'], 'echo_chambers/brexit/min_10.pkl', 'echo_chambers/brexit/graph'),
    (['capitalism'], 'echo_chambers/capitalism/min_10.pkl', 'echo_chambers/capitalism/graph'),
    (['nuclear-energy'], 'echo_chambers/nuclear-energy/min_10.pkl', 'echo_chambers/nuclear-energy/graph'),
]

exclude = ['76c3d895d71d8ad1a9c0df96e9e166d8148b118ca2f65b7bef69ff5d254a49de',
           '6b414b82ed587b0c07e8d98743906c499d35b60a9cadfab1ef50cb6404193b20',
           'd15ebb04ed2e8c68471009dcb7881b7fec6f3a84c20c4df1bbec699968fc60da',
           'f30d0cffb337c08a14f24bd269ff84e3877adf965149405a6e8f5f1664139b92',
           'b78e488903ed8ebf937d7fc1e3494414754faf84eb4baad8d16c1a3928dc3847',
           '365d662f290aad05f2c8e2a88a70914a186754f46caadde471dd436e459fc126',
           '4c27a433bcae7923662f317c6b6f373508be6ba0cf843127dd1e6299e5ad5499',
           '92f3133b701fbbd2bba2b20f233fa70faedbf4ce64d5c27da07c696803b4d378',
           'dbb0fc52ebd9fca22fb8d362cc65015e9a05aa8b71d822bd29234b86e89bd0e5',
           '9158db12ece9a3459709b33b939c20a551663f46cfd817634f58224d79efabde',
           'aeffb271b7abf6daec90bcb52628353845f260890b1cae856cdc1824ffe17be0',
           '1b15d4a6e207fcd2f92394dc36d6156bd788613efe758969157975495d4995d0',
           '3cd5415c01ec26da9e492bc1ad1bda6d4aab5837cb5403a67c4162693db4e9d1',
           '1f657581c89eb5aedf794655d6bb039697a3714da1846d4daf723c2499575146',
           '[deleted]']

def init_frames():
    for topic in TOPICS:
        try:
            print(topic)
            anno_users = [entry[0] for entry in annotated_user_map if entry[1] in topic[0]]
            print(len(anno_users))
            path = os.path.join(source_path, topic[0][0])
            init_dataset_from_db(PLATFORM, topic[0], 10, os.path.join(path, 'min_10.pkl'), added_users=anno_users)
        except Exception as e:
            print(e)
            continue

def visualize_graph(topic_tuple):
    plt.clf()
    loader = SocialGraphLoader(PLATFORM)
    d_frame = pd.read_pickle(topic_tuple[1], compression='infer')
    uids = list(d_frame['user_id'])
    print(len(uids))
    for uid in exclude:
        if uid in uids:
            uids.remove(uid)
    print(len(uids))
    loader.load_from_file(uids, topic_tuple[2])
    graph = loader.get_graph()
    graph.largest_connected_component()
    print('Start partitioning...')
    graph.partition_meta_algorithm('community_fluid')
    graph.color_nodes()
    graph.visualize()
    path = os.path.join(SOURCE_PATH, topic_tuple[0][0])
    plt.savefig(os.path.join(path, 'com_fluid.png'))
    with open(os.path.join(path, 'com_fluid_partition.pkl'), 'wb') as handle:
        pkl.dump(graph.partition, handle, protocol=pkl.HIGHEST_PROTOCOL)

    #Record metrics
    graph.record_metric('separability', 'separability')
    graph.record_metric('density', 'density')
    graph.record_user_degree()
    with open(os.path.join(path, 'com_fluid_metrics.pkl'), 'wb') as handle:
        pkl.dump(graph.metric_record, handle, protocol=pkl.HIGHEST_PROTOCOL)

def fill_with_posts(topic_tuple):
    fetch_posts_from_db(topic_tuple[1], 'REDDIT')

def annotate_stance_ground_truth(topic_tuple):
    annotator = GroundTruthStanceAnnotator(topic_tuple[1], 'gt_stance', topic_tuple[0])
    annotator.annotate()
    annotator.save_frame(topic_tuple[1])
    print(annotator.d_frame)

def main_pipeline(topic_tuple):
    print(topic_tuple[0])
    path = os.path.join(SOURCE_PATH, topic_tuple[0][0])
    d_frame = pd.read_pickle(topic_tuple[1], compression='infer')
    partition = pkl.load(open(os.path.join(path, 'com_fluid_partition.pkl'), 'rb'))
    #print(partition)
    cluster_stances = {}
    for index, row in d_frame.iterrows():
        if index in partition.keys():
            u_cluster = partition[index]
            u_stance = row['gt_stance']
            if u_stance == 'NOT_DEFINED':
                continue
        else:
            continue
        if u_cluster not in cluster_stances.keys():
            cluster_stances[u_cluster] = [u_stance]
        else:
            cluster_stances[u_cluster].append(u_stance)
    for cluster, stances in cluster_stances.items():
        print(str(cluster) + ": " + str(np.average(stances)) + " (n=" + str(len(stances)) + ")")


def build_socio_columns(annotator):
    annotator.annotate()
    res_id_col = []
    res_doc_col = []
    res_label_col = []

    for index, row in annotator.d_frame.iterrows():
        if row[annotator.label] == 'NOT_DEFINED':
            continue
        else:
            res_id_col.append(index)
            res_doc_col.append([post[2] for post in row['posts']])
            res_label_col.append(row[annotator.label])

    return res_id_col, res_doc_col, res_label_col

def annotate_socio_demographics(topic_tuple):
    res_path = topic_tuple[1].replace('.pkl', '_annotated.pkl')
    print(res_path)
    stance_annotator = GroundTruthStanceAnnotator(topic_tuple[1], 'gt_stance', topic_tuple[0])
    stance_annotator.annotate()
    stance_annotator.save_frame(res_path)
    gender_annotator = PredictionAnnotator(res_path, 'echo_chambers/updated_predictors/gender_linsvc.pkl', 'predicted_gender')
    gender_annotator.annotate()
    gender_annotator.save_frame(res_path)
    age_annotator = PredictionAnnotator(res_path, 'echo_chambers/updated_predictors/age_linsvc.pkl', 'predicted_age')
    age_annotator.annotate()
    age_annotator.save_frame(res_path)
    ideo_annotator = PredictionAnnotator(res_path, 'echo_chambers/updated_predictors/ideology_linsvc.pkl', 'predicted_ideology')
    ideo_annotator.annotate()
    ideo_annotator.save_frame(res_path)

def calc_results(topic_tuple):
    #print('\\textbf{' + str(topic_tuple[0][0]) + '}')
    frame_path = topic_tuple[1].replace('.pkl', '_annotated.pkl')
    path = os.path.join(SOURCE_PATH, topic_tuple[0][0])
    d_frame = pd.read_pickle(frame_path, compression='infer')
    res_writer = TopicResultWriter(d_frame, topic_tuple[0][0])

    #Load partition
    with open(os.path.join(path, 'com_fluid_partition.pkl'), 'rb') as handle:
        partition = pkl.load(handle)

    #Load metrics
    with open(os.path.join(path, 'com_fluid_metrics.pkl'), 'rb') as handle:
        metrics = pkl.load(handle)

    with open(os.path.join(path, 'com_fluid_metrics_exp.pkl'), 'rb') as handle:
        metrics_exp = pkl.load(handle)

    res_writer.add_graph_png(os.path.join(path, 'com_fluid.png'))
    res_writer.group_users_by_partition(partition)
    res_writer.add_stance_column('gt_stance', 'gt_stance')
    res_writer.add_stance_column('gt_stance', 'stance_weighted', weight_map=metrics['node_degree'])
    res_writer.add_graph_metric('d(c)', metrics['density'])
    res_writer.add_graph_metric('s(c)', metrics['separability'])
    res_writer.add_graph_metric('e(c)', metrics_exp['expansion'])
    res_writer.add_socio_column('predicted_gender', 'gender')
    res_writer.add_socio_column('regex_gender', 'gender_regex', regex_mode=True)
    res_writer.add_socio_column('predicted_age', 'age_group')
    res_writer.add_socio_column('regex_age', 'age_group_regex', regex_mode=True)
    res_writer.add_socio_column('predicted_ideology', 'ideology')
    res_writer.generate_latex_output_big(None)
    print('\\newpage')

def recalc_metrics(topic_tuple):
    loader = SocialGraphLoader(PLATFORM)
    d_frame = pd.read_pickle(topic_tuple[1], compression='infer')
    uids = list(d_frame['user_id'])
    print(len(uids))
    for uid in exclude:
        if uid in uids:
            uids.remove(uid)
    print(len(uids))
    loader.load_from_file(uids, topic_tuple[2])
    graph = loader.get_graph()
    graph.largest_connected_component()

    # Load partition
    path = os.path.join(SOURCE_PATH, topic_tuple[0][0])
    with open(os.path.join(path, 'com_fluid_partition.pkl'), 'rb') as handle:
        partition = pkl.load(handle)

    graph.partition = partition
    graph.color_nodes()
    graph.record_metric('separability', 'separability')
    graph.record_metric('density', 'density')
    print(graph.metric_record)
    graph.record_user_degree()
    with open(os.path.join(path, 'com_fluid_metrics.pkl'), 'wb') as handle:
        pkl.dump(graph.metric_record, handle, protocol=pkl.HIGHEST_PROTOCOL)
    graph.metric_record = {}
    graph.record_metric('expansion', 'expansion')
    print(graph.metric_record)
    with open(os.path.join(path, 'com_fluid_metrics_exp.pkl'), 'wb') as handle:
        pkl.dump(graph.metric_record, handle, protocol=pkl.HIGHEST_PROTOCOL)

def data_stats(tuple):
    print('----------')
    print(tuple[0])
    d_frame = pd.read_pickle(topic[1], compression='infer')
    num_posts = 0
    num_users = 0
    post_num_u = 0
    for index, row in d_frame.iterrows():
        if index not in user_list:
            user_list.append(index)
            post_num_u = post_num_u + len(row['posts'])
        num_posts += len(row['posts'])
        num_users += 1
    print(num_users)
    print(num_posts)
    return post_num_u

def annotate_regex(topic_tuple):
    res_path = topic_tuple[1].replace('.pkl', '_annotated.pkl')
    print(res_path)
    gender_annotator = RegexGenderAnnotator(res_path, 'regex_gender')
    gender_annotator.annotate()
    gender_annotator.save_frame(res_path)
    age_annotator = RegexAgeAnnotator(res_path, 'regex_age')
    age_annotator.annotate()
    age_annotator.save_frame(res_path)
    print(age_annotator.d_frame.columns)

def generate_cell_color(num):
    for index, rgb_val in enumerate([255 * color for color in cm.get_cmap('viridis', num).colors]):
        print('\\definecolor{color' + str(num) + str(index) + '}{RGB}{' + str(int(rgb_val[0])) + ',' + str(int(rgb_val[1])) +',' + str(int(rgb_val[2])) + '}')

def get_socio_correlation_values(topic_tuple, metric, socio_dim):
    frame_path = topic_tuple[1].replace('.pkl', '_annotated.pkl')
    path = os.path.join(SOURCE_PATH, topic_tuple[0][0])
    d_frame = pd.read_pickle(frame_path, compression='infer')
    res_writer = TopicResultWriter(d_frame, topic_tuple[0][0])

    # Load partition
    with open(os.path.join(path, 'com_fluid_partition.pkl'), 'rb') as handle:
        partition = pkl.load(handle)

    res_writer.group_users_by_partition(partition)
    x_vals = []
    print('Collecting x...')
    if metric == 'expansion':
        with open(os.path.join(path, 'com_fluid_metrics_exp.pkl'), 'rb') as handle:
            metrics_exp = pkl.load(handle)
        for com in sorted(metrics_exp['expansion']):
            x_vals.append(metrics_exp['expansion'][com])
    if metric == 'separability':
        with open(os.path.join(path, 'com_fluid_metrics.pkl'), 'rb') as handle:
            metrics = pkl.load(handle)
        for com in sorted(metrics['separability']):
            x_vals.append(metrics['separability'][com])
    print('Collecting y...')
    res_writer.add_socio_column(socio_dim, socio_dim)
    y_vals = res_writer.get_socio_correlation_values(socio_dim)
    return x_vals, y_vals

def get_stance_correlation_values(topic_tuple, metric, stance_key, std_mode=False):
    frame_path = topic_tuple[1].replace('.pkl', '_annotated.pkl')
    path = os.path.join(SOURCE_PATH, topic_tuple[0][0])
    d_frame = pd.read_pickle(frame_path, compression='infer')
    res_writer = TopicResultWriter(d_frame, topic_tuple[0][0])

    # Load partition
    with open(os.path.join(path, 'com_fluid_partition.pkl'), 'rb') as handle:
        partition = pkl.load(handle)

    res_writer.group_users_by_partition(partition)
    x_vals_prelim = []
    print('Collecting x...')
    with open(os.path.join(path, 'com_fluid_metrics.pkl'), 'rb') as handle:
        metrics = pkl.load(handle)
    if metric == 'expansion':
        with open(os.path.join(path, 'com_fluid_metrics_exp.pkl'), 'rb') as handle:
            metrics_exp = pkl.load(handle)
        for com in sorted(metrics_exp['expansion']):
            print(com)
            x_vals_prelim.append(metrics_exp['expansion'][com])
    if metric == 'separability':
        for com in sorted(metrics['separability']):
            print(com)
            x_vals_prelim.append(metrics['separability'][com])

    print('Collecting y...')
    res_writer.add_stance_column('gt_stance', 'gt_stance')
    res_writer.add_stance_column('gt_stance', 'stance_weighted', weight_map=metrics['node_degree'])
    print(x_vals_prelim)
    x_vals = []
    y_vals = []

    com_stances = []
    com_weights = []
    for com, stance_tuple in res_writer.column_data[stance_key].items():
        if stance_tuple[2] > 5:
            com_stances.append(stance_tuple[0])
            com_weights.append(stance_tuple[2])

    stance_mean = np.average(com_stances, weights=com_weights)

    for com, stance_tuple in res_writer.column_data[stance_key].items():
        print(com)
        print(stance_tuple)
        if stance_tuple[2] > 5:
            x_vals.append(x_vals_prelim[com])
            if not std_mode:
                y_vals.append(abs(stance_tuple[0]-stance_mean))
            else:
                y_vals.append(abs(stance_tuple[1]))
    return x_vals, y_vals

def calc_correlation(x, y):
    x = np.array(x)
    y = np.array(y)
    return scipy.stats.pearsonr(x, y)

def measure_socio_correlation(metric, socio_dim):
    print('--------------')
    print(metric)
    print(socio_dim)
    x_vals = []
    y_vals = []
    for topic in TOPICS:
        x_t , y_t = get_socio_correlation_values(topic, metric, socio_dim)
        x_vals.extend(x_t)
        y_vals.extend(y_t)
    print(x_vals)
    print(y_vals)
    print('Res: ' + str(calc_correlation(x_vals, y_vals)))

def measure_stance_correlation(metric, stance_key):
    print('--------------')
    print(metric)
    print(stance_key)
    x_vals = []
    y_vals = []
    for topic in TOPICS:
        x_t, y_t = get_stance_correlation_values(topic, metric, stance_key, std_mode=True)
        x_vals.extend(x_t)
        y_vals.extend(y_t)
    print(x_vals)
    print(y_vals)
    print('Res: ' + str(calc_correlation(x_vals, y_vals)))

def calc_pred_quality(topic_tuple):
    print('--------------')
    print(topic[0])
    frame_path = topic_tuple[1].replace('.pkl', '_annotated.pkl')
    path = os.path.join(SOURCE_PATH, topic_tuple[0][0])
    d_frame = pd.read_pickle(frame_path, compression='infer')
    res_writer = TopicResultWriter(d_frame, topic_tuple[0][0])
    with open(os.path.join(path, 'com_fluid_partition.pkl'), 'rb') as handle:
        partition = pkl.load(handle)

    res_writer.group_users_by_partition(partition)
    res_writer.add_socio_column('predicted_gender', 'predicted_gender')
    res_writer.add_socio_column('regex_gender', 'regex_gender')
    print(res_writer.column_data)
    print(res_writer.baselines)

for topic in TOPICS:
    calc_pred_quality(topic)

measure_socio_correlation('separability', 'predicted_gender')
measure_socio_correlation('separability', 'predicted_age')
measure_socio_correlation('separability', 'predicted_ideology')
measure_stance_correlation('separability', 'stance_weighted')
