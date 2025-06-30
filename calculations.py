import scipy.sparse as sp
import numpy as np

import bw2data as bd
import bw2analyzer as ba
from bw2calc.lca import LCA

from utils import *


def generate_matrices(lca, scenario) -> tuple[sp.csr_matrix, sp.csc_matrix]:
    """Genenerate new techosphere and/or biosphere matrices for the scenario."""

    # casting to lil_matrix is ~200x faster than using csr_matrix
    technosphere_matrix = sp.lil_matrix(lca.technosphere_matrix.copy())
    biosphere_matrix = sp.lil_matrix(lca.biosphere_matrix.copy())

    # set all values in scenario matrix to 0
    for from_key, to_key in scenario:
        if from_key[0] == "biosphere3":
            biosphere_matrix[lca.biosphere_dict[from_key], lca.activity_dict[to_key]] = 0
        else:
            if from_key == to_key:
                # don't update the diagonal to avoid empty rows in matrix
                continue
            technosphere_matrix[lca.product_dict[from_key], lca.activity_dict[to_key]] = 0
    return technosphere_matrix.tocsr(), biosphere_matrix.tocsc()


@timing
def get_scenario_matrices(lca, scenario_pairs: dict, direct_skips: set):
    biospheres = {("original",): lca.biosphere_matrix.copy()}
    hem_scenarios = {}

    # genenerate the matrices
    for scenario_name, scenario in scenario_pairs.items():

        technosphere, biosphere = generate_matrices(lca, scenario)
        hem_scenarios[scenario_name] = technosphere, biosphere, direct_skips

        sc_sector, sc_type = scenario_name
        biospheres[(sc_sector, "direct_remaining")] = biosphere

    return hem_scenarios, biospheres


def mlca(lca: LCA, calculation_setup, skip: set = None, progress: bool = False, convenience_print: bool = False,
         result_dict: dict = None, biospheres: dict = None) -> dict[tuple[str, tuple], dict[str, float]]:
    """Simple LCA calculation class to calculate scores for multiple activities and multiple methods.

    lca: LCA object
    calculation_setup: dict
        {"inv": list of dicts, "ia": list of methods}
    skip: set of activities to skip
    progress: bool
        print progress
    result_dict: dict
        dictionary to store results in
        format is: (activity key, biosphere) -> {method: score}
    biospheres: dict
        dict of biosphere matrices to use, if None, default is used
    """
    def do_lcia(_lca, _methods: list):
        """Perform LCIA calculation for all methods in list, return dict process contributions."""
        mthd_scores = {}
        for _method in _methods:
            _lca.switch_method(_method)
            _lca.lcia_calculation()

            ra, rp, rb = lca.reverse_dict()
            results = {
                ra[int(i)]: score
                for i, score in enumerate(lca.characterized_inventory.toarray().sum(axis=0))
            }

            mthd_scores[_method] = results
        return mthd_scores

    def print_progress(indicator):
        t_diff = time.time() - st_time
        lca_sec = max(i * n_mth * n_bio / t_diff, 1)
        remain = (n_act - i) * n_mth * n_bio / lca_sec
        if lca_sec < 1e3:
            lca_sec = f"{int(lca_sec)}"
        else:
            lca_sec = f"{round(lca_sec/1e3, 1)}k"
        print(f"\r {indicator} {lca_sec} LCA/s | "
              f"{round(i / n_act * 100, 1)}% | "
              f"ETA: {max(int(remain), 1)}s | "
              f"duration: {int(t_diff)}s", end="")

    # set data
    st_time = time.time()
    n_act = len(calculation_setup["inv"])
    n_mth = len(calculation_setup["ia"])

    if not skip:
        skip = set()

    if not result_dict:
        result_dict = {}

    # organize biospheres
    orig_biosphere = lca.biosphere_matrix.copy()
    if not biospheres:
        biospheres = {("original",): orig_biosphere}
    n_bio = len(biospheres.keys())

    # find total calculations
    n_tot = n_act*n_mth*n_bio
    if n_tot < 1e3:
        n_tot_str = str(n_tot)
    elif n_tot < 1e6:
        n_tot_str = f"{round(n_tot/1e3, 1)}k"
    else:
        n_tot_str = f"{round(n_tot / 1e6, 1)}M"

    # print convenience start info
    if convenience_print:
        if n_bio == 1:
            print(f"   run MLCA of {n_act} activities and {len(calculation_setup['ia'])} methods (n={n_tot_str})")
        else:
            print(f"   run MLCA of {n_act} activities, {len(calculation_setup['ia'])} "
                  f"methods and {n_bio} biospheres (n={n_tot_str})")

    # start actual calculation
    ind_c = 0  # used for progress indicator rotation
    pr_time = time.time()  # used for progress indicator timing
    methods = calculation_setup["ia"]
    skip_methods = {method: 0 for method in methods}
    for i, demand in enumerate(calculation_setup["inv"]):
        key = list(demand.keys())[0]

        # switch out biospheres for each calculation
        for bio_name, biosphere in biospheres.items():
            if key in skip:
                # shortcut the calculation if we know the result is 0 already
                result_dict[(key, bio_name)] = skip_methods
                continue
            # set new biosphere
            lca.biosphere_matrix = biosphere
            # set new inventory
            lca.redo_lci(demand)
            # calculate the scores
            result_dict[(key, bio_name)] = do_lcia(lca, methods)

        # print progress ~every second if enabled
        if progress and time.time() - pr_time > 1:
            print_progress(INDICATORS[ind_c])
            ind_c = (ind_c + 1) % 4
            pr_time = time.time()

    if convenience_print:
        # print final speed
        t_diff = time.time() - st_time
        print(f"\r > ran'mlca' in: {round(t_diff,4)}s | "
              f"{n_tot_str} LCAs finished @{int(round(n_tot / t_diff, 0))} LCA/s")
    else:
        print("\r", end="")  # fresh line

    # restore original biosphere
    lca.biosphere_matrix = orig_biosphere
    return result_dict


def techno_mlca(lca, calculation_setup, scenarios: dict, result_dict: dict = None):
    st_time = time.time()
    orig_technosphere = lca.technosphere_matrix.copy()
    orig_biosphere = lca.biosphere_matrix.copy()

    # find total calculations
    n_scn = len(scenarios)
    n_tot = len(calculation_setup["inv"]) * len(calculation_setup["ia"]) * n_scn
    if n_tot < 1e3:
        n_tot_str = str(n_tot)
    elif n_tot < 1e6:
        n_tot_str = f"{round(n_tot / 1e3, 1)}k"
    else:
        n_tot_str = f"{round(n_tot / 1e6, 1)}M"

    c = 1
    for sc_name, scenario in scenarios.items():
        sc_time = time.time()
        print(f" > run HEM scenario {c}/{n_scn}: '{sc_name[0]}'")
        new_technosphere, new_biosphere, skip = scenario
        biosphere_dict = {sc_name: new_biosphere}

        # re-initialize solver
        if hasattr(lca, "solver"):
            delattr(lca, "solver")
        lca.technosphere_matrix = new_technosphere
        lca.decompose_technosphere()

        # get new results
        result_dict = mlca(lca, calculation_setup, skip,
                           result_dict=result_dict, biospheres=biosphere_dict)

        sc_time = time.time() - sc_time
        print(f"   ran {c}/{n_scn} in {round(sc_time, 4)}s @{int(round(n_tot/n_scn/sc_time, 0))} LCA/s")
        c += 1

    t_diff = time.time() - st_time
    print(f"\r > ran'techno_mlca' in: {round(t_diff, 4)}s | "
          f"{n_tot_str} LCAs finished @{int(round(n_tot / t_diff, 0))} LCA/s")

    # restore original matrices
    lca.technosphere_matrix = orig_technosphere
    lca.biosphere_matrix = orig_biosphere

    return result_dict


@timing
def processing_scores(all_scores) -> pd.DataFrame:
    def contributions(df, col_name, top=3):
        ca = ba.contribution.ContributionAnalysis()

        df = df.copy()
        # df = df[[col_name, "product_name", "location"]]
        df = df[[col_name, "location"]]

        df = df.groupby(by=["location"]).sum()
        contributors = ca.sort_array(np.array(df[col_name]), limit=top)

        remainder = df[col_name].sum() - sum(contributors[:, 0])
        locs = [int(num) for num in contributors[:, 1]]
        df = df.iloc[locs]

        df_new = pd.DataFrame(data={col_name: remainder},
                              index=["remainder"])

        return pd.concat([df_new, df])

    # convert to a dataframe
    all_results = {}
    for (fu, scenario), results in all_scores.items():
        for method, scores in results.items():
            all_results[scenario[-1]] = scores
    df = pd.DataFrame(all_results)

    # drop rows where all values are 0
    df = df.loc[~(df==0).all(axis=1)]

    # calculate the target result
    df["target"] = df["original"] - df["remaining"]
    df["direct_target"] = df["original"] - df["direct_remaining"]

    # add human readable information about the processes
    activity_data = [bd.get_activity(key) for key in df.index]
    df["activity_data"] = activity_data
    #df["product_name"] = [act["reference product"] for act in activity_data]
    df["location"] = [act["location"] for act in activity_data]


    # sort
    df["sort_me"] = abs(df["original"])
    df = df.sort_values(by="sort_me", ascending=False)
    del df["sort_me"]

    df_remaining = contributions(df, "remaining")
    df_target = contributions(df, "target")
    contribution_dfs = [contributions(df, col_name) for col_name in
                        ["original", "remaining", "target", "direct_remaining", "direct_target"]]
    contribution_dfs = {"contributions": pd.concat(contribution_dfs, axis=1)}

    return df, contribution_dfs
