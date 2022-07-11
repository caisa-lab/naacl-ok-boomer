import numpy as np

gender_map = {0.0: "M",
              1.0: "F"}

age_map = {1.0: "$\\leq 30$",
           2.0: "$\\leq 45$",
           3.0: "$>45$"}

ideology_map = {
    0.0: "Con",
    1.0: "Mod",
    2.0: "Lib"
}

def rmse(error_vector):
    squared = [entry ** 2 for entry in error_vector]
    return (sum(squared) / len(error_vector)) ** 0.5

class TopicResultWriter(object):

    def __init__(self, d_frame, topic):
        self.topic = topic
        self.d_frame = d_frame
        self.graph_path = None
        self.grouped_users = {}
        self.column_data = {}
        self.partition = {}
        self.metric_metric = []
        self.baselines = {}

    def get_socio_correlation_values(self, key):
        rmse_vec = []
        for com in sorted(self.column_data[key]):
            print(com)
            com_errors = []
            for dim, percentage in self.column_data[key][com].items():
                com_errors.append(percentage-self.baselines[key][dim])
            rmse_vec.append(rmse(com_errors))
        return rmse_vec

    def add_graph_png(self, path):
        self.graph_path = path

    def group_users_by_partition(self, partition):
        for index, row in self.d_frame.iterrows():
            if not index in partition.keys():
                continue
            else:
                if partition[index] not in self.grouped_users.keys():
                    self.grouped_users[partition[index]] = [index]
                else:
                    self.grouped_users[partition[index]].append(index)
        self.partition = partition

    def add_stance_column(self, frame_label, column_name, weight_map=None):
        collector = {}
        weight_collector = {}
        for com in self.grouped_users.keys():
            collector[com] = []
            weight_collector[com] = []

        if weight_map is None:
            for index, row in self.d_frame.iterrows():
                if index not in self.partition.keys():
                    continue
                data = row[frame_label]
                if not data == 'NOT_DEFINED':
                    collector[self.partition[index]].append(data)

            res = {}
            for com, stances in collector.items():
                if len(stances) == 0:
                    mean_score = 0
                    std_score = 0
                    n_users = 0
                else:
                    mean_score =  np.mean(stances)
                    std_score = np.std(stances)
                    n_users = len(stances)
                res[com] = (mean_score, std_score, n_users)
            self.column_data[column_name] = res
        else:
            for index, row in self.d_frame.iterrows():
                if index not in self.partition.keys():
                    continue
                data = row[frame_label]
                if not data == 'NOT_DEFINED':
                    collector[self.partition[index]].append(data)
                    weight_collector[self.partition[index]].append(weight_map[index])

            res = {}
            for com, stances in collector.items():
                if len(stances) == 0:
                    mean_score = 0
                    std_score = 0
                    n_users = 0
                else:
                    mean_score =  np.average(stances, weights=weight_collector[com])
                    std_score = np.sqrt(np.average((stances-mean_score)**2, weights=weight_collector[com]))
                    n_users = len(stances)
                res[com] = (mean_score, std_score, n_users)
            self.column_data[column_name] = res

    def add_socio_column(self, frame_label, column_name, regex_mode=False):
        collector = {}
        baseline_collector = {}
        for com in self.grouped_users.keys():
            collector[com] = {}

        for index, row in self.d_frame.iterrows():
            if index not in self.partition.keys():
                continue
            socio_label = row[frame_label]
            if socio_label == "NOT_DEFINED":
                continue
            if socio_label not in collector[self.partition[index]].keys():
                collector[self.partition[index]][socio_label] = 1
            else:
                collector[self.partition[index]][socio_label] += 1

            if socio_label not in baseline_collector.keys():
                baseline_collector[socio_label] = 1
            else:
                baseline_collector[socio_label] += 1
        res = {}
        #TODO: Fix missing entries
        for com, socios in collector.items():
            total = sum(socios.values())
            res[com] = {}
            for socio, amount in socios.items():
                if not regex_mode:
                    res[com][socio] = amount/total
                else:
                    res[com][socio] = ((amount/total), amount)

        self.column_data[column_name] = res
        baseline_res = {}
        print(sum(baseline_collector.values()))
        for index, summed in baseline_collector.items():
            baseline_res[index] = summed/sum(baseline_collector.values())
        self.baselines[column_name] = baseline_res

    def add_graph_metric(self, column_name, metric_map):
        self.column_data[column_name] = metric_map

    def generate_latex_output_big(self, res_path):
        latex_string = "\\begin{figure*}[hbt!]\n"
        latex_string += "\\centering\n"
        latex_string += "\\small\n"
        latex_string += "\\includegraphics[scale=0.65]{" + self.graph_path + "}\n"
        latex_string += "\\begin{tabular}{| l | c | c | c | c | c | c | c | c | c |}\n \
                \\hline\n \
                \\multicolumn{4}{|c|}{{\\large Cluster}} & \\multicolumn{3}{c|}{{\\large Metrics}} & \\multicolumn{3}{c|}{{\\large Sociodemographics}} \\\\ \n \
                \\hline \n"
        latex_string += " & \\textbf{\#Users} & \\textbf{stance}& \\begin{minipage}[t]{0.2\columnwidth}\\textbf{weighted \\newline stance \\newline}" \
                        "\\end{minipage} & \\textbf{d(c)} & \\textbf{s(c)} & \\textbf{e(c)} & \\textbf{Gender} &" \
                        "\\textbf{Age} &" \
                        "\\textbf{Ideology}  \\\\"
        latex_string += "\\hline \n"

        columns = ['gt_stance', 'stance_weighted', 'd(c)', 's(c)', 'e(c)', 'gender', 'age_group' , 'ideology']

        for com_index, com in enumerate(sorted(self.grouped_users)):
            line = '\\cellcolor{color' + str(max(self.grouped_users.keys()) + 1) + str(com_index) + '} ' + str(com)
            num_users = len(self.grouped_users[com_index])
            line += " & "
            line += str(num_users)
            for column in columns:
                data = self.column_data[column]
                line += " & "
                if isinstance(data[com], tuple):
                    if column == 'gt_stance':
                        scale = 0.20
                        line += "\\begin{minipage}[t]{" + str(scale) + "\columnwidth}"
                        line += "\\diameter: " +str(round(float(data[com][0]), 3))
                        line += " \\newline "
                        line += "Std: " + str(round(float(data[com][1]), 3))
                        line += " \\newline "
                        line += "\\#Users: " + str(round(float(data[com][2]), 3))
                        line += "\\end{minipage}"
                    if column == 'stance_weighted':
                        scale = 0.20
                        line += "\\begin{minipage}[t]{" + str(scale) + "\columnwidth}"
                        line += "$\\diameter$: " +str(round(float(data[com][0]), 3))
                        line += " \\newline "
                        line += "Std: " + str(round(float(data[com][1]), 3))
                        line += " \\newline "
                        line += "\\end{minipage}"
                elif isinstance(data[com], dict):
                    if column == 'gender':
                        scale = 0.16
                    elif column == 'age_group':
                        scale = 0.25
                    elif column == 'ideology':
                        scale = 0.27
                    else:
                        scale= 0.15
                    line += "\\begin{minipage}[t]{" + str(scale) + "\columnwidth}"
                    for socio in sorted(data[com]):
                        if column == 'gender' or column == 'gender_regex':
                            socio_label = gender_map[socio]
                        if column == 'age_group' or column == 'age_group_regex':
                            socio_label = age_map[socio]
                        if column == 'ideology':
                            socio_label = ideology_map[socio]
                        if not isinstance(data[com][socio], tuple):
                            line += socio_label + ": " + str(round(data[com][socio], 3)) + " \\newline "
                        else:
                            #TODO: Fix amount
                            line += socio_label + ": " + str(round(data[com][socio][0], 3)) + " \\newline "
                    line += "\\end{minipage}"
                else:
                    line += str(round(data[com], 3))
            line += "\\\\"
            line += "\\hline \n"
            latex_string += line


        latex_string += "\\end{tabular} \n \
                \\caption{\\centering " + self.topic + " } \n \\end{figure*}"

        print(latex_string)
