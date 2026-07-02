from calculation_settings import methods, mining_hem, location_hem
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
    ('ecoinvent-3.11-cutoff', '9e580072f69b141c3254ab82a0e56c07'),  # copper, cathode | market for copper, cathode | GLO
]

HEM_SCENARIOS = [
    mining_hem,
    location_hem
]

CONTRIBUTION_FIELDS = [
    "reference product",
    "location"
]

# Get a tree structure of CPC data
CPC_tree = get_cpc_tree()

# Set the current project
bd.projects.set_current(PROJECT)

# Load the database
if DB_NAME not in bd.databases:
    raise ValueError(f"Database {DB_NAME} not found in project {PROJECT}")

# Create a dataframe of the database
db_df = load_bw_2_pd(DB_NAME)
db_df = unpack_classifications(db_df, CLASSIFICATIONS)

# Iterate over all sets of FU, HEM scenario and contribution field
for functional_unit, scenario, contribution_field in zip(FUS, HEM_SCENARIOS, CONTRIBUTION_FIELDS):

    df = db_df.copy()

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
    if scenario["type"] == "cpc":
        df, scenarios = identify_cpc_scenario(df, scenario["name"], CPC_tree, assign_other=False)
    else:
        df, scenarios = identify_non_cpc_scenario(df, scenario)

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
    scores = processing_scores(all_scores, contribution_field=contribution_field)
    print("+ Exporting file")
    export_df_to_xlsx(scores, f"export {str(bd.get_activity(functional_unit))} {scenario['name']}.xlsx")
    print("+ Done")
