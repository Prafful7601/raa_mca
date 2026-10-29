"""Loading and serialising TabFact tables (# -delimited CSVs)."""
import functools
import pandas as pd


def load_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="#", dtype=str, keep_default_na=False, na_values=[])
    return df


@functools.lru_cache(maxsize=4096)
def load_table_cached(csv_path: str) -> pd.DataFrame:
    return load_table(csv_path)


def df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def table_id_to_csv_path(table_id: str, tables_dir: str) -> str:
    return f"{tables_dir}/{table_id}"


def load_generic_table(path_or_buffer) -> pd.DataFrame:
    """Standard comma-delimited CSV loader for user-supplied tables (API/demo),
    as opposed to TabFact's '#'-delimited source files."""
    return pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False, na_values=[])
