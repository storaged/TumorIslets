import time
import statistics
from itertools import groupby

from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse import csgraph
import pandas as pd
from matplotlib import collections as mc
from collections import Counter
import logging
from collections import defaultdict
from Cell import Cell, Marker
from scipy.spatial import distance
import pylab as plt
import numpy as np
import multiprocessing
import Concave_Hull
from scipy.spatial import Delaunay

import linecache
import os
import tracemalloc

logging.basicConfig(level=logging.INFO)


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def divide_chunks(my_list, number_of_chunks, sorted_list=False):
    # looping till length l
    if sorted_list:
        total = 0
        idx_from = 0
        while total < len(my_list):
            section = my_list[idx_from:len(my_list):number_of_chunks]
            total += len(section)
            idx_from += 1
            yield section
    else:
        for i in range(0, len(my_list), number_of_chunks):
            yield my_list[i:i + number_of_chunks]


def create_cells_parallel(initial_structure, indices):
    tmp_position_to_cell_mapping = dict()
    tmp_id_to_position_mapping = dict()
    for i in indices:
        cell_1 = Cell(initial_structure.loc[i], i)
        tmp_id_to_position_mapping[i] = (cell_1.x, cell_1.y)
        tmp_position_to_cell_mapping[(cell_1.x, cell_1.y)] = cell_1
    return tmp_id_to_position_mapping, tmp_position_to_cell_mapping


def _determine_margin_helper(number, boundary):
    start = None
    last = None
    margin_positions = []
    margin_edge_sequence = []
    position_to_component = defaultdict(list)

    for px, py in zip(*boundary.xy):
        position = (px, py)
        margin_positions.append(position)
        if not start:
            start = position
        position_to_component[position] = number
        if not last:
            last = position
            continue

        margin_edge_sequence.append([last, position])
        last = position

    return margin_edge_sequence, margin_positions, position_to_component


def determine_margin_parallel(points_for_concave_hull, number, alpha=20):

    if len(points_for_concave_hull) <= 4:
        return None, None

    try:
        concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
    except:
        print(points_for_concave_hull)
        print("Empty edges for component: {}", number)

    if not edge_points:
        return [], [], defaultdict(list)
    elif type(concave_hull.boundary).__name__ == "LineString":
        return _determine_margin_helper(boundary=concave_hull.boundary, number=number)
    else:
        _inv_mar_seq_tmp = []
        _inv_mar_tmp = []
        _position_to_component = defaultdict(list)
        for geom in concave_hull.geoms:
            margin_edge_sequence, margin_positions, position_to_component = _determine_margin_helper(
                boundary=geom.boundary,
                number=number
            )
            _inv_mar_seq_tmp.extend(margin_edge_sequence)
            _inv_mar_tmp.extend(margin_positions)
            _position_to_component.update(position_to_component)
        return _inv_mar_seq_tmp, _inv_mar_tmp, _position_to_component


def _parallel_helper_margin_detection(islets, alpha):
    tmp_invasive_margins_sequence = dict()
    tmp_invasive_margins = dict()
    position_to_component = defaultdict(list)

    for idx, (islet_number, points_for_concave_hull) in enumerate(islets):
        _seq, _cells, _position_to_component = determine_margin_parallel(points_for_concave_hull, islet_number,
                                                                         alpha=alpha)
        tmp_invasive_margins_sequence[islet_number] = _seq
        tmp_invasive_margins[islet_number] = _cells
        position_to_component.update(_position_to_component)

    return tmp_invasive_margins_sequence, tmp_invasive_margins, position_to_component


class Graph(object):

    def __init__(self, initial_structure=None, max_dist=50, mode='connectivity', run_parallel=False):
        """
        Initializes a graph object.
        If no dictionary or None is given, an empty dictionary will be used.
        Possible mode in {‘connectivity’, ‘distance’}
        :type max_dist: int
        """
        self._connected_components_positions = defaultdict(list)
        self._CK_neighbour_graph = None
        self._connected_components = None
        self._invasive_margins = defaultdict(list)
        self._invasive_margins_sequence = defaultdict(list)
        self._id_to_position_mapping = dict()
        self._position_to_cell_mapping = dict()
        self.max_dist = max_dist
        self._position_to_component = dict()

        if type(initial_structure).__name__ == "dict":

            if initial_structure is None:
                initial_structure = {}
            self._graph_dict = initial_structure

        elif type(initial_structure).__name__ == "DataFrame":

            points_list = list(zip(initial_structure["nucleus.x"], initial_structure["nucleus.y"]))

            start = time.process_time()
            self._graph_dict = radius_neighbors_graph(points_list,
                                                      self.max_dist,
                                                      mode=mode,
                                                      include_self=True,
                                                      n_jobs=int(0.75 * multiprocessing.cpu_count()))
            print("Runtime, radius_neighbors_graph: " + str(time.process_time() - start))

            cx = self._graph_dict.tocoo()

            keys_total = 0
            self._id_to_position_mapping = {}
            self._position_to_cell_mapping = {}
            start = time.process_time()

            unique_keys_id = {i for i in cx.row}

            ########
            ## BASIC MULTIPROCESSING APPROACH
            ########
            if run_parallel:
                n = 20000
                idx_subsets = list(divide_chunks(list(unique_keys_id), n))
                start = time.process_time()
                with multiprocessing.Pool(processes=int(0.75 * multiprocessing.cpu_count()),
                                          maxtasksperchild=1000) as pool:

                    params = [(initial_structure, idx_subset) for idx_subset in idx_subsets]
                    results = pool.starmap_async(create_cells_parallel, params)

                    for p in results.get():
                        tmp_id_to_position_mapping, tmp_position_to_cell_mapping = p
                        self._id_to_position_mapping.update(tmp_id_to_position_mapping)
                        self._position_to_cell_mapping.update(tmp_position_to_cell_mapping)

            else:
                for i in unique_keys_id:
                    keys_total += 1

                    cell_1 = Cell(initial_structure.loc[i], i)
                    self._id_to_position_mapping[i] = (cell_1.x, cell_1.y)
                    self._position_to_cell_mapping[(cell_1.x, cell_1.y)] = cell_1

            print("Runtime, (Graph.__init__) {}, keys {}: ".format(time.process_time() - start,
                                                                   len(self._id_to_position_mapping)))

    def edges(self, vertice):
        """ returns a list of all the edges of a vertice"""
        return self._graph_dict[vertice]

    def all_vertices(self):
        """ returns the vertices of a graph as a set """
        return set(self._graph_dict.keys())

    def all_vertices_coords_only(self):
        """ returns the vertices of a graph as a set """
        return [cell.position for cell in self._graph_dict.keys()]

    def all_vertices_as_lists(self):
        return list(map(list, self._graph_dict.keys()))

    def all_edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def get_edges(self, vertex):
        if vertex in self._graph_dict:
            return self._graph_dict[vertex]
        else:
            return set()

    def parse_and_add_vertex(self, record, connect=True):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.   """

        cell = Cell(record)

        if cell not in self._graph_dict:
            self._graph_dict[cell] = []
        if connect:
            for cell_old in self._graph_dict.keys():
                self.add_edge(cell, cell_old)

    def add_vertex(self, vertex, connect=True):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.   """

        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []
        if connect:
            for vertex_old in self._graph_dict.keys():
                self.add_edge(vertex, vertex_old)

    def add_edge(self, vertex1, vertex2, directed=True):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!  """

        if type(vertex1).__name__ == "Cell":

            to_add = [(vertex1, vertex2)]
            if not directed: to_add.append((vertex2, vertex1))

            for x, y in to_add:
                self._graph_dict[x].append(y)

        # The ELSE part is to be removed once Cell class/type is tested properly
        else:
            if vertex1 != vertex2 and distance.euclidean(vertex1, vertex2) < self.max_dist:
                for x, y in [(vertex1, vertex2), (vertex2, vertex1)]:
                    if x in self._graph_dict:
                        self._graph_dict[x].append((x, y))
                    else:
                        self._graph_dict[x] = []
                        self._graph_dict[x].append((x, y))

    def connected_components(self, mode='connectivity'):
        """value: n_components, labels"""
        if not self._connected_components:
            start_time = time.process_time()
            print("Start computing connected_components().")
            ck_positive_x, ck_positive_y, idxs = list(
                zip(*[(val[0], val[1], key) for key, val in self._id_to_position_mapping.items() if
                      self._position_to_cell_mapping[val].marker_is_active(Marker.CK)])  # activities['CK']])
            )
            ck_cells = list(zip(ck_positive_x, ck_positive_y))

            self._CK_neighbour_graph = radius_neighbors_graph(ck_cells,
                                                              self.max_dist,
                                                              mode=mode,
                                                              include_self=False,
                                                              n_jobs=int(0.75 * multiprocessing.cpu_count()))

            self._connected_components = csgraph.connected_components(self._CK_neighbour_graph)

            tmp = np.array(list(self._id_to_position_mapping.keys()))
            tmp[:] = -1
            tmp[np.array(idxs)] = self._connected_components[1]
            self._connected_components = (self._connected_components[0], tmp)

            print("Dictionary for connected_components().")
            mid_time = time.process_time()

            self._connected_components_positions = {k: [self._id_to_position_mapping[pos]
                                                        for pos, val in g] for k, g in
                                                    groupby(sorted(enumerate(self._connected_components[1]),
                                                                   key=lambda x: x[1]),
                                                            key=lambda x: x[1])
                                                    }
            print("Runtime, dictionary connected_components(): {}. ".format(
                round(time.process_time() - mid_time, 3)))

            print("Runtime, connected_components(): {}. ".format(
                round(time.process_time() - start_time, 3)))
        return self._connected_components

    def characterize_CK_islets(self):
        ctr = Counter(self.connected_components()[1])

        ctr = {k: [v] for k, v in ctr.items()}
        return pd.DataFrame.from_dict(ctr)

    def select_CK_component_IDs(self, number):
        return np.where(self.connected_components()[1] == number)[0]

    def select_CK_component_points(self, number):
        cells_ids = np.where(self.connected_components()[1] == number)[0]
        return [position for id_, position in self._id_to_position_mapping.items() if id_ in cells_ids]

    def get_CK_component_by_number(self, number):
        return self._connected_components_positions[number]

    def determine_all_margins(self, alpha=20, min_islet_size=10, run_parallel=True):

        start_time = time.process_time()
        mid_time = start_time

        islet_sizes = self.characterize_CK_islets()

        if run_parallel:

            start_time = time.process_time()
            self.connected_components()

            print("Preparing data with components for parallelization.")

            selected_islets = [(idx, islet_sizes.at[0, idx], self.get_CK_component_by_number(idx))
                               for idx in islet_sizes
                               if islet_sizes.at[0, idx] >= min_islet_size and idx != -1]
            selected_islets = [(idx, component)
                               for idx, size, component in sorted(selected_islets, key=lambda x: x[1], reverse=True)]

            print("Runtime, data prepared: {}. ".format(
                round(time.process_time() - start_time, 3)))

            number_of_cpu = int(0.75 * multiprocessing.cpu_count())
            number_of_islets = len(selected_islets)
            n = int(number_of_islets / number_of_cpu) + 1
            if n <= 2:
                n = int(number_of_islets / 10) + 1
            else:
                n = number_of_cpu

            islet_subsets = list(divide_chunks(selected_islets, n, sorted_list=True))

            print("[MULTIPROCESSING] Calculate {} margins.\nTrying to invoke {} tasks on {} cpu".format(
                number_of_islets, n, number_of_cpu
            ))

            with multiprocessing.Pool(processes=32, maxtasksperchild=32) as pool:
                params = [(islets, alpha) for islets in islet_subsets]

                ## APPLY ASYNC
                # results = [pool.apply_async(self._parallel_helper_margin_detection, p) for p in params]
                ## print("Len of results: " + str(len(results)))
                # for p in results:
                #    inv_mar_seq, inv_mar, position_to_component = p.get()
                #    self._invasive_margins_sequence.update(inv_mar_seq)
                #    self._invasive_margins.update(inv_mar)
                #    self._position_to_component.update(position_to_component)

                ## MAP ASYNC
                results = pool.starmap_async(_parallel_helper_margin_detection, params)
                for p in results.get():
                    inv_mar_seq, inv_mar, position_to_component = p
                    self._invasive_margins_sequence.update(inv_mar_seq)
                    self._invasive_margins.update(inv_mar)
                    self._position_to_component.update(position_to_component)

        else:
            selected_islets = [(idx, islet_sizes.at[0, idx])
                               for idx in islet_sizes
                               if islet_sizes.at[0, idx] >= min_islet_size and idx != -1]
            selected_islets = [(idx, size)
                               for idx, size in sorted(selected_islets, key=lambda x: x[1], reverse=True)]

            # WITHOUT MULTIPROCESSING
            for islet_number, islet_size in selected_islets:
                print("Determine margin for islet number: " + str(islet_number) + " of size " + str(
                    islet_size))

                inv_mar_seq, inv_mar, position_to_component = self.determine_margin(islet_number, alpha=alpha)
                self._invasive_margins_sequence[islet_number] = inv_mar_seq
                self._invasive_margins[islet_number] = inv_mar
                self._position_to_component.update(position_to_component)

                print("Runtime, determine_margin(): {}. ".format(
                    round(time.process_time() - mid_time, 3)))
                mid_time = time.process_time()

        print(("Runtime, determine_all_margins(): {}. " +
               "Number of margins analyzed: {}").format(
            round(time.process_time() - start_time, 3),
            len(selected_islets)))

    def determine_margin(self, number, alpha=20):
        points_for_concave_hull = self.select_CK_component_points(number)

        if len(points_for_concave_hull) <= 4:
            return None, None

        # concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
        try:
            concave_hull, edge_points = Concave_Hull.alpha_shape(points_for_concave_hull, alpha=alpha)
        except:
            print(points_for_concave_hull)
            print("Empty edges for component: {}", number)

        if not edge_points:
            return [], [], defaultdict(list)
        elif type(concave_hull.boundary).__name__ == "LineString":
            return _determine_margin_helper(boundary=concave_hull.boundary, number=number)
        else:
            _inv_mar_seq_tmp = []
            _inv_mar_tmp = []
            _position_to_component = defaultdict(list)
            for geom in concave_hull.geoms:
                margin_edge_sequence, margin_positions, position_to_component = _determine_margin_helper(
                    boundary=geom.boundary,
                    number=number
                )
                _inv_mar_seq_tmp.extend(margin_edge_sequence)
                _inv_mar_tmp.extend(margin_positions)
                _position_to_component.update(position_to_component)
            return _inv_mar_seq_tmp, _inv_mar_tmp, _position_to_component

    def get_invasive_margin(self, number):
        cells_positions = self._invasive_margins[number]
        cx = self._graph_dict.tocoo()

        invasive_margin = []

        for i in cells_positions:
            cell_i = self._position_to_cell_mapping[i]
            invasive_margin.append(cell_i)
            cell_i_id = cell_i.id  # list(self._id_to_cell_mapping)[list(self._id_to_cell_mapping.values()).index(cell_i)]
            for j in self._graph_dict[cell_i_id, :].indices:
                invasive_margin.append(self._position_to_cell_mapping[self._id_to_position_mapping[j]])

        # return invasive_margin
        return list(set(invasive_margin))

    def characterize_invasive_margins(self, numbers=None, path_to_save=None, save_plot=True, display_plots=False):

        start_time = time.process_time()
        mid_time = start_time
        if numbers is None:
            numbers = [key for key, val in self._invasive_margins.items() if key != -1 and val != []]

        if type(numbers).__name__ == "int":
            numbers = [numbers]

        print("Runtime, characterize_invasive_margins():compute numbers: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        all_phenotypes = []
        margin_number = []
        for number in numbers:
            inv_mar = [cell.phenotype_label for cell in self.get_invasive_margin(number)]
            all_phenotypes.extend(inv_mar)
            margin_number.extend([number] * len(inv_mar))

        print("Runtime, characterize_invasive_margins():iterate over numbers: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        data = {
            'Margin Number': margin_number,
            'Cell Type': all_phenotypes}

        pal = self._create_color_dictionary(set(all_phenotypes))

        df = pd.DataFrame(data)
        ax = pd.crosstab(df['Margin Number'], df['Cell Type']).apply(lambda r: r / r.sum() * 100, axis=1)
        tmp = pd.crosstab(df['Margin Number'], df['Cell Type'])
        if path_to_save:
            tmp.to_csv(path_to_save + "-description-margin-{}.csv".format(self.max_dist))

        print("Runtime, characterize_invasive_margins():to_csv: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        if save_plot:
            total_cells = tmp.transpose().sum().tolist()

            ax_1 = ax.plot.bar(figsize=(16, 9), stacked=True, rot=0, color=pal)

            # display(ax)

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       fancybox=True, shadow=True, ncol=7, title="Gene Marker")

            plt.xlabel('Margin Number')
            plt.ylabel('Percent Distribution')

            for idx, rec in enumerate(ax_1.patches):
                height = rec.get_height()
                _row = idx % tmp.shape[0]
                _col = idx // tmp.shape[0]
                ax_1.text(rec.get_x() + rec.get_width() / 2,
                          rec.get_y() + height / 2,
                          # "{:.0f}% ({:.0f})".format(height, tmp.iat[_row,_col]) if height != 0 else " ",
                          "{:.0f}%".format(height) if height >= 5 else " ",
                          ha='center',
                          va='bottom',
                          rotation=90)

            labels = [item.get_text() + " (" + str(total_cells[idx]) + ")" for idx, item in
                      enumerate(ax_1.get_xticklabels())]

            ax_1.set_xticklabels(labels)
            plt.xticks(rotation=90)
            plt.savefig(path_to_save + "-description-margin-{}.pdf".format(self.max_dist))
            if display_plots:
                plt.show()
            print("Runtime, characterize_invasive_margins():plot {}.".format(
                round(time.process_time() - start_time, 3)))

    @staticmethod
    def _create_color_dictionary(all_phenotypes):

        no_pheno = len(all_phenotypes)

        ck_phenotypes = [v for v in all_phenotypes if "CK" in v]
        non_ck_phenotypes = [v for v in all_phenotypes if not "CK" in v and v != "neg"]

        no_ck_phenotypes = len(ck_phenotypes)
        no_non_ck_phenotypes = len(non_ck_phenotypes)

        greens_palette = [plt.cm.get_cmap("Greens_r", no_non_ck_phenotypes + 1)(i) for i in
                          range(no_non_ck_phenotypes + 1)]
        reds_palette = [plt.cm.get_cmap("YlOrRd_r", no_ck_phenotypes + 1)(i) for i in range(no_ck_phenotypes + 1)]

        ck_colors = dict(zip(ck_phenotypes, reds_palette[:no_ck_phenotypes]))
        non_ck_colors = dict(zip(non_ck_phenotypes, greens_palette[:no_non_ck_phenotypes]))

        color_dict_phenotypes = dict(ck_colors, **non_ck_colors)

        color_dict_phenotypes["neg"] = (0, 0.7, 1, 1)

        return color_dict_phenotypes

    def plot(self, subset=None, componentNumber=None, pMarginBorder=True, pVert=True,
             pIsletEdges=True, pMarginEdges=True, pOuterEdges=True, pOuterCells=True, addLabels=True,
             isletAlpha=0.075, marginIsletAlpha=0.5, marginAlpha=0.075, display_plots=False,
             s=5, figsize=(16, 12), path_to_save="tmp", verbose=False):

        print("Graph.plot(): START")
        start_time = time.process_time()
        mid_time = start_time

        if not subset and not componentNumber:
            subset = self._id_to_position_mapping.keys()
        if componentNumber is not None:
            subset = self.select_CK_component_IDs(componentNumber).tolist()

        plt.rcParams['figure.figsize'] = figsize

        cells_to_plot = list(map(self._position_to_cell_mapping.get,
                                 list(map(self._id_to_position_mapping.get, subset))
                                 ))

        print("Runtime, Graph.plot():compute cells_to_plot: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        if verbose:
            for cell in cells_to_plot:
                print(cell)

        xs = np.array([cell.x for cell in cells_to_plot])
        ys = np.array([cell.y for cell in cells_to_plot])
        phenotypes = np.array([cell.phenotype_label for cell in cells_to_plot])

        print("Runtime, Graph.plot():compute arrays of data: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        fig, ax = plt.subplots()

        lines = {"Islet-Islet": [], "Margin-Islet": [], "Margin-Border": [], "Margin": []}
        other_cells_phenotype = []
        other_cells_x = []
        other_cells_y = []

        # Gathering lines

        # Lines (edges) describing a path of the islet border
        if pMarginBorder:
            components_list = []
            if type(componentNumber).__name__ == "int":
                components_list = [componentNumber]
            elif componentNumber is None:
                components_list = self._invasive_margins_sequence.keys()
            for cmpNum in components_list:
                _inv_mar_seq = self._invasive_margins_sequence[cmpNum]
                lines["Margin-Border"].extend(_inv_mar_seq)

                if _inv_mar_seq:
                    (lab_x1, lab_y1), (lab_x2, lab_y2) = _inv_mar_seq[0]
                    ax.text((lab_x1 + lab_x2) / 2, (lab_y1 + lab_y2) / 2,
                            cmpNum, fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="pink", ec="cyan", lw=1, alpha=0.5),
                            ha='center', va='center')

        print("Runtime, Graph.plot():pMarginBorder: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        # Lines (edges) connecting cells within CK-islet and neighbouring not CK-cells
        if pIsletEdges or pOuterEdges:

            cx = self._graph_dict.tocoo()
            for i in subset:
                cell_i = self._position_to_cell_mapping[self._id_to_position_mapping[i]]

                for j in self._graph_dict[i, :].indices:
                    cell_j = self._position_to_cell_mapping[self._id_to_position_mapping[j]]
                    line = (cell_i.position,
                            cell_j.position)
                    if j in subset:
                        lines["Islet-Islet"].append(line)

                    else:
                        other_cells_x.append(cell_j.x)
                        other_cells_y.append(cell_j.y)
                        other_cells_phenotype.append(cell_j.phenotype_label)
                        lines["Margin-Islet"].append(line)

            lines["Islet-Islet"] = [(x, y) for x, y in set(lines["Islet-Islet"])]
            lines["Margin-Islet"] = [(x, y) for x, y in set(lines["Margin-Islet"])]
            other_cells_x = np.array(other_cells_x)
            other_cells_y = np.array(other_cells_y)
            other_cells_phenotype = np.array(other_cells_phenotype)

        print("Runtime, Graph.plot():pIsletEdges or pOuterEdges: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        if pOuterCells and (componentNumber is not None):
            cx = self._graph_dict.tocoo()
            for i, j, _ in zip(cx.row, cx.col, cx.data):
                cell_j = self._position_to_cell_mapping[self._id_to_position_mapping[j]]
                cell_i = self._position_to_cell_mapping[self._id_to_position_mapping[i]]

                if cell_i.position in self._position_to_component and \
                        self._position_to_component[cell_i.position] == componentNumber and \
                        not cell_j.marker_is_active(Marker.CK):  # activities["CK"]:
                    other_cells_x.append(cell_j.x)
                    other_cells_y.append(cell_j.y)
                    other_cells_phenotype.append(cell_j.phenotype_label)
            other_cells_x = np.array(other_cells_x)
            other_cells_y = np.array(other_cells_y)
            other_cells_phenotype = np.array(other_cells_phenotype)

        print("Runtime, Graph.plot():pOuterCells: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()
        # Lines (edges) conecting cells within the margin limits (< 50um from the islet border)

        if pMarginEdges:
            components_list = []
            if type(componentNumber).__name__ == "int":
                components_list = [componentNumber]
            elif componentNumber is None:
                components_list = self._invasive_margins_sequence.keys()

            for id_ in components_list:
                margin_cells = self.get_invasive_margin(id_)
                if margin_cells:
                    _polygon_points = [cell.position for cell in margin_cells]

                    tri = Delaunay(_polygon_points)

                    thresh = self.max_dist
                    short_edges = set()
                    dev_null = set()
                    new_tri = []
                    for tr in tri.vertices:
                        segment_count = 0
                        for i in range(3):
                            edge_idx0 = tr[i]
                            edge_idx1 = tr[(i + 1) % 3]
                            p0 = _polygon_points[edge_idx0]
                            p1 = _polygon_points[edge_idx1]

                            if distance.euclidean(p1, p0) <= thresh:
                                segment_count += 1
                                if segment_count > 1:
                                    new_tri.append(tr)
                                    break

                            if (edge_idx1, edge_idx0) in dev_null or (edge_idx1, edge_idx0) in short_edges:
                                continue

                            if distance.euclidean(p1, p0) <= thresh:
                                short_edges.add((edge_idx0, edge_idx1))
                            else:
                                dev_null.add((edge_idx0, edge_idx1))
                    lines["Margin"] = [[_polygon_points[i], _polygon_points[j]] for i, j in short_edges]

                    tri_x = np.array([x for x, y in _polygon_points])
                    tri_y = np.array([y for x, y in _polygon_points])
                    pts = np.zeros((len(_polygon_points), 2))
                    pts[:, 0] = tri_x
                    pts[:, 1] = tri_y
                    plt.tripcolor(pts[:, 0], pts[:, 1], np.array(len(new_tri) * [0.2]),
                                  triangles=new_tri, alpha=0.2, edgecolors='k')
                    del dev_null

        print("Runtime, Graph.plot():pMarginEdges: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        lines_properties_dict = {"Islet-Islet": {"color": "Red", "alpha": isletAlpha,
                                                 "linestyle": (0, (5, 10)), "draw": pIsletEdges},
                                 "Margin-Islet": {"color": "Green", "alpha": marginIsletAlpha,
                                                  "linestyle": (0, (3, 5, 1, 5)), "draw": pOuterEdges},
                                 "Margin-Border": {"color": "Purple", "alpha": marginAlpha,
                                                   "linestyle": 'solid', "draw": pMarginBorder},
                                 "Margin": {"color": "Purple", "alpha": marginAlpha,
                                            "linestyle": (0, (5, 10)), "draw": pMarginEdges}}

        for interaction in lines.keys():
            if lines_properties_dict[interaction]["draw"]:
                line_collection = mc.LineCollection(lines[interaction],
                                                    linestyle=lines_properties_dict[interaction]["linestyle"],
                                                    colors=lines_properties_dict[interaction]["color"],
                                                    label=interaction,
                                                    linewidth=0.5,
                                                    alpha=lines_properties_dict[interaction]["alpha"])
                ax.add_collection(line_collection)
                ax.autoscale()
                ax.margins(0.1)
                ax.grid(True)

        print("Runtime, Graph.plot():for interaction in lines.keys():: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        from matplotlib.lines import Line2D

        def make_proxy(color, marker, label, **kwargs):
            if marker == 'o':
                return Line2D([0], [0], color='w', markerfacecolor=color, marker=marker, label=label, **kwargs)
            else:
                return Line2D([0], [0], color=color, label=label, **kwargs)

        proxies = [make_proxy(properties["color"],
                              marker=None, label=name, alpha=properties["alpha"],
                              linestyle=properties["linestyle"])
                   for name, properties in lines_properties_dict.items()]

        first_legend = ax.legend(handles=proxies, loc='upper right')
        ax.add_artist(first_legend)

        # Gathering points
        set_of_all_phenotypes = set(phenotypes).union(set(other_cells_phenotype))

        color_dict_phenotypes = self._create_color_dictionary(set_of_all_phenotypes)

        if pVert:

            for phenotype in set(phenotypes):
                to_plot = np.where((phenotypes == phenotype))[0]
                ax.scatter(xs[to_plot], ys[to_plot], alpha=isletAlpha,
                           color=color_dict_phenotypes[phenotype],
                           label=phenotype, s=s, marker=".")

                # to_plot = np.where((phenotypes == phenotype) & (1 == on_margins))[0]
                # ax.scatter(xs[to_plot], ys[to_plot], alpha=isletAlpha,
                #           color=color_dict_phenotypes[phenotype],
                #           label=phenotype, s=s, marker="o")
                # to_plot = np.where((phenotypes == phenotype) & (0 == on_margins))[0]
                # ax.scatter(xs[to_plot], ys[to_plot], alpha=isletAlpha,
                #           color=color_dict_phenotypes[phenotype],
                #           label=phenotype, s=s, marker="X")
            logging.debug(("Points collections: \n" +
                           "Islet: {};").format(len(cells_to_plot)))

            if pOuterCells:
                for phenotype in set(other_cells_phenotype):
                    to_plot = np.where(other_cells_phenotype == phenotype)[0]
                    ax.scatter(other_cells_x[to_plot], other_cells_y[to_plot],
                               color=color_dict_phenotypes[phenotype], alpha=marginIsletAlpha,
                               label=phenotype, s=s)
            logging.debug(("Points collections: \n" +
                           "Outer (nonCK): {};").format(len(other_cells_x)))

            proxies = [make_proxy(color, marker='o', label=name, markersize=s) for name, color in
                       color_dict_phenotypes.items()]
            second_legend = ax.legend(handles=proxies, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                                      fancybox=True, shadow=True, ncol=7, title="Gene Marker")

            ax.add_artist(second_legend)

        print("Runtime, Graph.plot():pVert: {}.".format(
            round(time.process_time() - mid_time, 3)))
        mid_time = time.process_time()

        start_time = time.process_time()
        ax.autoscale()
        plt.savefig(path_to_save + "-margin-{}.pdf".format(self.max_dist))
        if display_plots:
            plt.show()

        print(("Runtime, plot(): ({}).\n" +
               "EDGES -- Islet-Islet: {}; Margin-Margin: {}; Margin-Islet: {}, Margin Border: {}\n" +
               "VERTICES -- Inner cells: {}; Outer cells: {}").format(
            round(time.process_time() - start_time, 3),
            len(lines["Islet-Islet"]), len(lines["Margin"]),
            len(lines["Margin-Islet"]), len(lines["Margin-Border"]),
            len(xs), len(other_cells_x)))

    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices    """

        edges = set()
        for cell in self._graph_dict:
            for neighbour in self._graph_dict[cell]:
                if type(neighbour).__name__ == "tuple":
                    if neighbour not in edges:
                        edges.add(neighbour)
                else:
                    edges.add((cell, neighbour))
        return edges

    def __iter__(self):
        self._iter_obj = iter(self._graph_dict)
        return self._iter_obj

    def __next__(self):
        """ allows us to iterate over the vertices """
        return next(self._iter_obj)

    def __str__(self):
        res: str = "Cells: \n"

        if len(self._graph_dict) < 20:
            for k in self._graph_dict:
                res += str(k) + ":\n"
                res += "\tNeighbours (" + str(len(self._graph_dict[k])) + ")\n"
                for neighbour in self._graph_dict[k]:
                    res += "[-> " + str(neighbour) + "], "
                res += "\n"
        else:
            res += str(self._graph_dict) + ":\n"
            lengths = [len(v) for v in self._graph_dict.values()]
            res += "Mean number of neighbours: " + str(round(statistics.fmean([1, 2, 4]), 2)) + ",\n"
            res += "Median number of neighbours: " + str(round(statistics.median([1, 2, 4]), 2)) + "."

        return res
