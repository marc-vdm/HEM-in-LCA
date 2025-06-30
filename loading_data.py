from typing import Iterable, Any

import bw2data as bd

from utils import *


#
# Brightway data
#
@timing
def load_bw_2_pd(db_name: str) -> pd.DataFrame:
    """Read data from Brighway, generate keys, sort on keys, drop un-needed columns."""
    print(" - run:'load_bw_2_pd' ...", end="")
    df = pd.DataFrame(bd.Database(db_name))
    df["key"] = df.loc[:, ["database", "code"]].apply(tuple, axis=1)
    df.sort_values(by="key", inplace=True)  # sort by key to make order deterministic

    keep_cols = ["production amount", "reference product", "name", "unit", "location", "key", "classifications"]
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]
    return df


#
# CPC classification data
#
def get_cpc_tree() -> dict:
    """Generate an entry for every class of the CPC and store its path.

    this file is from https://unstats.un.org/unsd/classifications/Econ/cpc
    stored locally under path variable below
    the file is sorted and structured such that each sub-class of the previous has 1 character more in column
    'code', that means each super-class is already seen before we get to the sub-class
    we use that as a feature to create the 'tree path'

    Returns
    -------
            tree_data: keys are str of classification:name, values are the tree path consisting of keys
    """
    path = os.path.join(os.getcwd(),
                        "CPC_Ver_2_1_english_structure.txt")
    df = pd.read_csv(path, dtype=str)

    tree_data = {}
    last_super = tuple()
    last_super_depth = 0
    for idx, row in df.iterrows():
        cls, name = row  # cls is the number classification, name is the proper name
        current_depth = len(cls)  # we measure the depth by the length of cls
        key = f'{cls}:{name}'

        if current_depth > last_super_depth:
            # this is a sub-class at a deeper level as the last entry we read
            path = tuple(list(last_super) + [key])  # create a tuple of the tree path
        elif current_depth <= last_super_depth:
            # this is a (sub-)class at a same or higher level than the last entry we read
            depth = last_super_depth - current_depth + 1  # find how many entries to clip of the path
            path = tuple(list(last_super)[:-depth] + [key])  # create a tuple of the tree path

        tree_data[key] = path  # add the treepath to the key in dict
        last_super = path  # add as last_super

        # take the last class level, split on ':' and take the length of the class as depth
        last_super_depth = len(last_super[-1].split(":")[0])
    return tree_data


@timing
def unpack_classifications(df: pd.DataFrame, systems: list) -> pd.DataFrame:
    """Unpack classifications column to a new column for every classification system in 'systems'.

    Will return dataframe with added column.
    """
    def unpacker(classifications: list, system: str) -> list:
        """Iterate over all 'c' lists in 'classifications'
        and add those matching 'system' to list 'x', when no matches, add empty string.
        If 'c' is not a list, add empty string.

        Always returns a list 'x' where len(x) == len(classifications)
        """
        system_classifications = []
        for c in classifications:
            result = ""
            if not isinstance(c, (list, tuple, set)):
                system_classifications.append(result)
                continue
            elif isinstance(c, tuple):
                if c[0] == system:
                    result = c[1]
            else:
                for s in c:
                    if s[0] == system:
                        result = s[1]
                        result = result.replace(": ", ":")
                        break
            system_classifications.append(result)  # result is either "" or the classification
        return system_classifications

    classifications = list(df['classifications'].values)
    system_cols = []
    for i, system in enumerate(systems):
        system_cols.append(unpacker(classifications, system))
    # creating the DF rotated is easier so we do that and then transpose
    unpacked = pd.DataFrame(system_cols, columns=df.index, index=systems).T

    # Finally, merge the df with the new unpacked df using indexes
    df = pd.merge(
        df, unpacked, how="inner", left_index=True,
        right_index=True, sort=False
    )
    return df


#
# retrieve scenario key pairs
#
@timing
def identify_scenario(df: pd.DataFrame, scenarios, path_dict, assign_other=True) -> tuple[pd.DataFrame, list]:
    """Add new column to df 'scenarios'.

    Supports scenarios, either as list of strings, as tuples of strings or as dicts of strings
    (can be tuples of tuples with eventually strings).
    If tuple of strings, the first string is the main scenario, the rest are sub-scenarios.
    The sub-scenarios are checked first, then the 'main' scenario.
    """
    def check_scenarios(_row: str, _scenarios: Iterable, _path_dict) -> tuple[str, bool]:
        _scen = ""
        _found = False
        for _scenario in _scenarios:
            if (isinstance(_scenario, str)
                    and _scenario in _path_dict.get(_row, "No classification")):
                # the scenario is str and found in the path dict
                return _scenario, True
            elif isinstance(_scenario, tuple):
                # scenario is a tuple, skip the first and recursively check the rest
                _scen, _found = check_scenarios(_row, _scenario[1:], _path_dict)
                if not _found:
                    # was not found in recursion, check first element
                    _scen, _found = check_scenarios(_row, [_scenario[0]], _path_dict)
                if _found:
                    return _scen, True
            elif isinstance(_scenario, dict):
                # scenario is a dict, aggregate all underlying
                _scen, _found = check_scenarios(_row, list(_scenario.values())[0], _path_dict)
                if _found:
                    return list(_scenario.keys())[0], True
        return _scen, _found

    def scen_add(_scenarios: Iterable) -> list:
        new_scens = []
        for _scenario in _scenarios:
            if isinstance(_scenario, str):
                new_scens.append(_scenario)
            elif isinstance(_scenario, tuple):
                new_scens = new_scens + scen_add(_scenario)
            elif isinstance(_scenario, dict):
                new_scens.append(list(_scenario.keys())[0])
        return new_scens

    col = df["CPC"].to_list()
    scenario_col = []
    for row in col:
        scen, found = check_scenarios(row, scenarios, path_dict)
        if found:
            scenario_col.append(scen)
        else:  # no match was found
            if assign_other:
                scenario_col.append("Other")
            else:
                scenario_col.append("No Scenario Assigned")

    # create new scenarios set that has scenarios that are actually present
    new_scenarios = scen_add(scenarios)
    if assign_other:
        new_scenarios.append("Other")
    # drop any unused scenarios
    sc_cl = set(scenario_col)
    for sc in [sc for sc in new_scenarios if sc not in sc_cl]:
        new_scenarios.remove(sc)

    # add column to df
    df["scenarios"] = scenario_col

    return df, new_scenarios


@timing
def get_scenario_data(df: pd.DataFrame, scenarios: list[str], progress=False) -> tuple[
    dict[tuple[str, str], list[tuple[Any, Any]]], set[Any]]:
    """Create pairs of keys defining exchanges that are part of each given scenario.
    """
    def print_progress(indicator):
        t_diff = time.time() - st_time
        print(f"\r {indicator} run:'get_scenario_data' | "
              f"{i+1}/{len(scenarios)} | "
              f"duration: {time_format(t_diff)}", end="")

    # time tracking
    st_time = time.time()
    pr_time = st_time
    ind_c = 0

    scenario_pairs = {}
    direct_skips = set()
    for i, scenario in enumerate(scenarios):
        # filter the DF so it only contains the activities that match the scenario
        filtered_df = df[df.apply(lambda
                                      row: scenario == row["scenarios"],
                                  axis=1)]
        if len(filtered_df) == 0:
            # this scenario does not return data
            continue

        hem_exch_pairs = []

        for idx, row in filtered_df.iterrows():
            act = bd.get_activity(row["key"])

            if len(act.technosphere()) == 0 and len(act.biosphere()) == 0:
                # we can skip activities with no input exchanges for all calculations
                direct_skips.add(row["key"])
                continue

            for exchange in act.upstream():
                # pair is from current activity to other activity (current act is input)
                hem_exch_pairs.append((row["key"], exchange["output"]))
            for exchange in act.biosphere():
                # pair is from biosphere to current activity (current act is output)
                hem_exch_pairs.append((exchange["input"], row["key"]))

            # print progress ~every second if enabled
            if progress and time.time() - pr_time > 1:
                indicators = INDICATORS
                print_progress(indicators[ind_c])
                ind_c = (ind_c + 1) % len(indicators)
                pr_time = time.time()

        scenario_pairs[(scenario, "remaining")] = hem_exch_pairs

    return scenario_pairs, direct_skips
