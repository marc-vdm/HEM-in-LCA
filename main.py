from calculation_settings import methods, mining_hem
from calculations import *
from loading_data import *

import bw2data as bd
import bw2calc as bc
import time

start_time = time.time()

PROJECT = "ei311 hem"
DB_NAME = "ecoinvent-3.11-cutoff"
CLASSIFICATIONS = ["CPC"]

FUS = [
    ('ecoinvent-3.11-cutoff', '9e580072f69b141c3254ab82a0e56c07'),  # copper, cathode | market for copper, cathode | GLO
]
HEM_SCENARIOS = [
    mining_hem,
]

for functional_unit, scenario in zip(FUS, HEM_SCENARIOS):

    # Set the current project
    bd.projects.set_current(PROJECT)

    CPC_tree = get_cpc_tree()

    # Load the database
    if DB_NAME not in bd.databases:
        raise ValueError(f"Database {DB_NAME} not found in project {PROJECT}")

    df = load_bw_2_pd(DB_NAME)
    df = unpack_classifications(df, CLASSIFICATIONS)

    # create calculation setup
    reference_flows = [{functional_unit: 1}]
    calculation_setup = {"inv": reference_flows,
                         "ia": methods}
    # add the FU amounts as column
    refs = {list(d.keys())[0]: d[list(d.keys())[0]] for d in reference_flows}
    df["fu_amount"] = df["key"].map(refs)

    # initialize LCA object
    lca = bc.lca.LCA(demand=calculation_setup["inv"][0], method=calculation_setup["ia"][0])
    lca.lci(factorize=True)

    # generate scenario matrices
    df, scenarios = identify_scenario(df, scenario, CPC_tree, assign_other=False)
    scenario_pairs, direct_skips = get_scenario_data(
        df,
        scenarios=scenarios,
        progress=True)
    hem_scenarios, biospheres = get_scenario_matrices(lca, scenario_pairs, direct_skips)
    all_scores = {}

    # calculate the default and 'direct' scores
    print("+ Calculating default and direct scores")
    new_scores = mlca(lca, calculation_setup, biospheres=biospheres, skip=direct_skips)
    all_scores.update(new_scores)

    print("+ Calculating HEM scores")
    new_scores = techno_mlca(lca, calculation_setup, scenarios=hem_scenarios)
    all_scores.update(new_scores)

    print("+ Processing results")
    scores = processing_scores(all_scores)
    export_df_to_xlsx(scores, f"export {str(bd.get_activity(functional_unit))} {scenario}.xlsx")
