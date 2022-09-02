# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import sys
import argparse

import pandas as pd

from Graph import *
# os.system("taskset -p 0xff %d" % os.getpid())

import linecache
import os
import tracemalloc


def parse_my_arguments(args):
    parser = argparse.ArgumentParser(description='Process Islet Graph Analysis arguments.')
    parser.add_argument('--n_head', dest='n_head', type=int, nargs='?',
                        help='number of records from IF Panel to visualize'),
    parser.add_argument('--max_cells', dest='max_cells', type=int, nargs='?',
                        help='max number of records in IF data')
    parser.add_argument('--max_panels', dest='max_panels', type=int, nargs='?',
                        help='number of panels to visualize')
    parser.add_argument('--alpha', dest='alpha', type=int, nargs='?', default=30,
                        help='Alpha shape parameter')
    parser.add_argument('--display_plots', dest='display_plots', type=bool, nargs='?',
                        const=True, default=False,
                        help='Should we display plots on fly?')
    parser.add_argument('--skip_plots', dest='skip_plots', type=bool, nargs='?',
                        const=True, default=False,
                        help='Should we compute plots?')
    parser.add_argument('--parallel', dest='parallel', type=bool, nargs='?',
                        const=True, default=False,
                        help='Should we run it in multiprocessing mode?')
    parser.add_argument('--labels', dest='labels', type=bool, nargs='?',
                        const=True, default=False,
                        help='Should we add layer with labels onto plots?')
    parser.add_argument('--neighbour_dist', dest='neighbour_dist', type=int, nargs='?', default=50,
                        help='number of panels to visualize')
    parser.add_argument('--testing', dest='testing', type=str, nargs='?',
                        default='none',
                        help='number of panels to visualize')

    return parser.parse_args(args)


def get_next_instruction():
    int_val = None

    while True:
        try:
            int_val = int(input('Enter the component number of interest (or -1: quit; 999: full panel):'))
        except ValueError:
            print("Input value needs to be a number (N: component number, -1: quit, 999: full panel).")
            continue
        break

    return int_val


def main():
    import socket

    from datetime import date
    from pathlib import Path

    today = date.today()

    tracemalloc.start()
    args = parse_my_arguments(sys.argv[1:])
    machine_name = socket.gethostname()

    if machine_name == 'MacBook-Pro-Krzysiek.local':
        my_dir = "/Users/krzysiek/PROJEKTY_NAUKOWE/IMMUCAN/tsv-files/"
    elif machine_name == 'amor':
        my_dir = "/home/krzysiek/IMMUCAN/tsv-files/"
    elif machine_name == 'rudy':
        my_dir = "/home/kgogolewski/IMMUCAN/tsv-files/"
    else:
        my_dir = "./tsv-files"

    # filename = '/Users/krzysiek/PROJEKTY_NAUKOWE/IMMUCAN/tsv-files/IMMU-NSCLC-0179-FIXT-01-IF1-01.tsv.gz'
    # inv_marg_file = filename.replace("tsv-files", "invasive_margins")

    from os import listdir
    from os.path import isfile, join

    if args.testing == 'local':
        only_files = [my_dir + "IMMU-NSCLC-0179-FIXT-01-IF1-01.tsv.gz"]
    elif args.testing == 'memory':
        only_files = [my_dir + "IMMU-NSCLC-0182-FIXT-01-IF1-01.tsv.gz"]
    else:
        only_files = [my_dir + f for f in listdir(my_dir) if isfile(join(my_dir, f)) if "tsv.gz" in f]
        if args.max_panels is not None and args.max_panels < len(only_files):
            only_files = only_files[:args.max_panels]

    for filename in only_files:
        inv_marg_file = filename.replace("tsv-files", "invasive_margins_" + today.strftime("%y_%m_%d")).replace(
            ".tsv.gz", "")
        Path(my_dir.replace("tsv-files", "invasive_margins_" + today.strftime("%y_%m_%d"))).mkdir(exist_ok=True)

        cells_df = pd.read_csv(filename, sep="\t")
        max_cells = cells_df.shape[0] + 1 if args.max_cells is None else args.max_cells

        if cells_df.shape[0] < max_cells:
            print("[INFO] Analysis of the file: {} with {} cells).".format(filename, cells_df.shape[0]))

            cells_df = cells_df if args.n_head is None else cells_df.head(args.n_head)
            graph_new = Graph(cells_df, max_dist=args.neighbour_dist, run_parallel=args.parallel)
            graph_new.determine_all_margins(alpha=args.alpha, run_parallel=args.parallel)

            graph = Graph(cells_df)
            graph.connected_components()

            print("How many:", graph._connected_components[0])
            # graph.slicing_all_graph("component_slices/slicing10up_0193.csv")

            # graph_new.compute_all_components_slices(steepness=5)
            print("START: graph_new.compute_all_components_slices()")
            graph_new.compute_all_components_slices(alpha=args.alpha, steepness=5)
            print("END: graph_new.compute_all_components_slices()")

            # graph_new.characterize_invasive_margins(path_to_save=inv_marg_file,
            #                                        display_plots=args.display_plots,
            #                                        save_plot=True, plot_labels=args.labels)

            steep_regions_dataframes = []

            for id, position in graph_new._id_to_position_mapping.items():
                data = {"x-axis": position[0],
                            "y-axis": position[1],
                            "component_number": graph_new._position_to_ck_component[position],
                            "component_margin_number": graph_new._position_to_component[position],
                            "margin_in_component_number": graph_new._position_to_margin_in_component[position][1],
                            "steep_region_number": graph_new._position_to_steep_region[position],
                            "cell_type": graph_new._position_to_cell_mapping[position].phenotype_label}

                steep_regions_dataframes.append(data)

            df_to_save = pd.DataFrame(steep_regions_dataframes)
            panel_desc_file = filename.replace("tsv-files", "panels_description").replace(".tsv.gz", ".csv")
            df_to_save.to_csv(panel_desc_file)

            if args.skip_plots:
                continue

            if args.testing == "local":
                print(Counter(graph_new._position_to_component.values()).keys())
                print(Counter(graph_new._position_to_component.values()).values())
                val_int = get_next_instruction()

                while val_int != -1:
                    if val_int == 999:
                        graph_new.plot(  # subset = graph_new.select_CK_component_IDs(17).tolist(),
                            # componentNumber=val,
                            s=10, verbose=False, pMarginBorder=True, path_to_save=inv_marg_file,
                            pIsletEdges=False, pMarginEdges=True, pOuterEdges=False,
                            pOuterCells=True, pVert=True, plot_labels=args.labels,
                            isletAlpha=0.2, marginIsletAlpha=0.2, marginAlpha=1, pSlices=True,
                            display_plots=args.display_plots)
                    else:
                        graph_new.plot(  # subset = graph_new.select_CK_component_IDs(17).tolist(),
                            componentNumber=val_int,
                            s=10, verbose=False, pMarginBorder=True, path_to_save=inv_marg_file,
                            pIsletEdges=False, pMarginEdges=True, pOuterEdges=False,
                            pOuterCells=True, pVert=True, plot_labels=args.labels, pSlices=True,
                            isletAlpha=0.2, marginIsletAlpha=0.2, marginAlpha=1, display_plots=args.display_plots)

                    val_int = get_next_instruction()

            else:
                graph_new.plot(  # subset = graph_new.select_CK_component_IDs(17).tolist(),
                    # componentNumber=val,
                    s=10, verbose=False, pMarginBorder=True, path_to_save=inv_marg_file,
                    pIsletEdges=False, pMarginEdges=False, pOuterEdges=False,
                    pOuterCells=True, pVert=True, plot_labels=args.labels,
                    isletAlpha=0.1, marginIsletAlpha=0.2, marginAlpha=1, display_plots=args.display_plots)
                # graph_new._CK_neighbour_graph







        else:
            print("[INFO] Skip file: {} because it has {} cells (max set to: {}).".format(filename, cells_df.shape[0],
                                                                                          max_cells))
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)


if __name__ == '__main__':
    main()
