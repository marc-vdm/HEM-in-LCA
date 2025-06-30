from functools import wraps
import time
import os

import pandas as pd

INDICATORS = ["●∙∙∙", "∙●∙∙", "∙∙●∙", "∙∙∙●", "∙∙●∙", "∙●∙∙"]

def timing(f):
    """timing wrapper for functions.

    From https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f"\r > ran:'{f.__name__}' in: {time_format(te-ts)}")
        return result
    return wrap


def time_format(t):
    if t < 10:
        return f"{round(t, 1)}s"
    elif t <= 60:
        return f"{int(round(t, 0))}s"
    elif t < 3600:
        m = int(round(t / 60, 0))
        s = int(round(t % 60, 0))
        s = f"0{s}" if s < 10 else f"{s}"
        return f"{m}m{s}s"
    else:
        h = int(round(t / 3600, 0))
        t = t - (h * 3600)
        m = int(round(t / 60, 0))
        m = f"0{m}" if m < 10 else f"{m}"
        s = int(round(t % 60, 0))
        s = f"0{s}" if s < 10 else f"{s}"
        return f"{h}h{m}m{s}s"


def print_mlca_progress(indicator, t_diff, i, n_mth, n_bio, n_act, lca_sec=False, prepend=""):
    lca_sec = lca_sec if lca_sec else max(i * n_mth * n_bio / t_diff, 1)
    if t_diff < 5:  # if progressed time is still low, don't estimate ETA
        remain = "estimating..."
    else:
        remain = ((n_act - i) * n_mth * n_bio) / lca_sec
        remain = time_format(max(remain, 1))

    if lca_sec < 1e3:
        lca_sec = f"{int(lca_sec)}"
    else:
        lca_sec = f"{round(lca_sec/1e3, 1)}k"
    print(f"\r {indicator} {prepend}"
          f"{lca_sec} LCA/s | "
          f"{round(i / n_act * 100, 1)}% | "
          f"Scenario ETA: {remain} | "
          f"duration: {time_format(t_diff)}", end="")


def export_df_to_xlsx(dfs, file_name):
    def write_excel_tab(_df, sheet_name):
        with pd.ExcelWriter(full_path, engine="openpyxl", mode="a") as writer:
            workBook = writer.book
            try:
                workBook.remove(workBook[sheet_name])
            except:
                pass  # this tab apparently doesn't exist
            finally:
                _df.to_excel(writer, sheet_name=sheet_name, index=True, header=True)

    df, contribution_dfs = dfs

    file_name = file_name.replace("'", "")
    file_name = file_name.replace(",", "")
    file_name = file_name.replace(":", "-")
    full_path = os.path.join(os.getcwd(), file_name)

    df.to_excel(full_path, sheet_name="results")
    write_excel_tab(df, "results")
    for col_name, _df in contribution_dfs.items():
        write_excel_tab(_df, col_name)
