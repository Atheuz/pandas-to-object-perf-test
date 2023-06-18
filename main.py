"""Test creating object from DataFrame rows."""
import random
import statistics
import uuid

import numpy as np
import pandas as pd
import perfplot
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["figure.dpi"] = 144


# Helper functions
def unwrap(series, keys_to_keep):
    """Convert a pd Series to dict, with keys_to_keep."""
    d = series.to_dict()
    d2 = {k: d[k] for k in keys_to_keep}
    return d2


def tuple_unwrap(t, keys_to_keep):
    """Convert a namedtuple to dict, with keys_to_keep."""
    d = t._asdict()
    d2 = {k: d[k] for k in keys_to_keep}
    return d2


# Classes and data creation.
# The order of the create_node function arguments. This is important, as * args unpacking must be in this order, ** kwargs unpacking can be in any order.
LOOKUP = ["b", "a", "f", "e", "d", "c", "g"]


class Node:
    """Node class."""

    def __init__(self, node_id, b, a, f, c, d, e, g):
        """Initialize with some data, in a specific order."""
        self.node_id = node_id
        self.b = b
        self.a = a
        self.f = g
        self.c = c
        self.d = d
        self.e = e
        self.g = g

    def __repr__(self):
        """String representation of the node."""
        return f"{self.b=}, {self.a=}, {self.f=}, {self.c=}, {self.d=}, {self.e=}, {self.g=}"

    def __eq__(self, other):
        """Check if a node is equivalent to another node, based on all fields except node_id."""
        if not isinstance(other, Node):
            return False
        return self.b == other.b and self.a == other.a and self.f == other.f and self.c == other.c and self.d == other.d and self.e == other.e and self.g == other.g

    def __hash__(self):
        """Hash node based on all fields. Convert fields to str."""
        # necessary for instances to behave sanely in dicts and sets.
        return hash((self.node_id, str(self.b), str(self.a), str(self.f), str(self.c), str(self.d), str(self.e), str(self.g)))


def create_node(b, a, f, e, d, c, g):
    """Function to create a node, passing in a UUID, and keeping the rest of the args the same."""
    return Node(str(uuid.uuid4()), b, a, f, c, d, e, g)


def create_node_ignored_args(b, a, f, e, d, c, g, *args, **kwargs):
    """Function to create a node, passing in a UUID, and keeping the rest of the args the same.

    Allow other args and kwargs to be passed as well, but will be ignored.
    """
    return Node(str(uuid.uuid4()), b, a, f, c, d, e, g)


def random_letter_numbers(choices, start=1, end=100):
    """Generate a list of tuples from choices (as keys), and random.randint(start,end) as the value."""
    inner_choices = list(choices)
    random.shuffle(inner_choices)
    return [(x, random.randint(start, end)) for x in inner_choices]


def random_dict():
    """Create a random dict that contains 6 scalar fields, and 1 non-scalar field."""
    choices = list("abcdefg")
    subset_n = 5
    random.shuffle(choices)
    scalars = random_letter_numbers(choices[:-1])
    obj = [(choices[-1], {"subset": [random.randint(1, 10) for _ in range(subset_n)]})]
    return dict(scalars + obj + [("trash", "field")])


def random_df(n):
    """Create a DataFrame from n random_dicts."""
    return pd.DataFrame([random_dict() for _ in range(n)])


# Functions to test.
def index_df_apply(df):
    """Use apply, get the fields indexing using LOOKUP.

    Use as args in create_node (using *).
    """
    nodes = df.apply(lambda row: create_node(*row[LOOKUP]), axis=1)
    return [node for node in nodes]


def index_df_apply_kwargs(df):
    """Use apply, get the fields indexing using LOOKUP.

    Use as kwargs in create_node (using **).
    """
    nodes = df.apply(lambda row: create_node(**row[LOOKUP]), axis=1)
    return [node for node in nodes]


def index_df_apply_kwargs_ignored_args(df):
    """Use apply.

    Use all fields as kwargs in create_node_ignored_args (using **).
    """
    nodes = df.apply(lambda row: create_node_ignored_args(**row), axis=1)
    return [node for node in nodes]


def index_dict_apply(df):
    """Use apply, convert the row to a dict, get the fields using unwrap fn according to LOOKUP.

    Use as kwargs in create_node (using **).
    """
    nodes = df.apply(lambda row: create_node(**unwrap(row, LOOKUP)), axis=1)
    return [node for node in nodes]


def index_dict_apply_ignored_args(df):
    """Use apply, convert the row to a dict.

    Use all fields as kwargs in create_node_ignored_args (using **).
    """
    nodes = df.apply(lambda row: create_node_ignored_args(**row.to_dict()), axis=1)
    return [node for node in nodes]


def index_dict_iterrows(df):
    """Loop over iterrows, convert row to a dict, get the fields using unwrap fn according to LOOKUP.

    Use as kwargs in create_node (using **).
    """
    nodes = []
    for _, row in df.iterrows():
        nodes.append(create_node(**unwrap(row, LOOKUP)))
    return nodes


def index_dict_iterrows_comprehension(df=None):
    """List comprehension over iterrows, convert row to a dict, get the fields using unwrap fn according to LOOKUP.

    Use as kwargs in create_node (using **).
    """
    nodes = [create_node(**unwrap(row, LOOKUP)) for _, row in df.iterrows()]
    return nodes


def index_dict_iterrows_comprehension_ignored_args(df=None):
    """List comprehension over iterrows, convert row to a dict.

    Use all fields as kwargs in create_node_ignored_args (using **).
    """
    nodes = [create_node_ignored_args(**row.to_dict()) for _, row in df.iterrows()]
    return nodes


def index_series_iterrows(df):
    """Loop over iterrows, get the fields using indexing using LOOKUP.

    Use as args in create_node (using *).
    """
    nodes = []
    for _, row in df.iterrows():
        nodes.append(create_node(*row[LOOKUP]))
    return nodes


def index_series_iterrows_comprehension(df):
    """List comprehension iterrows, get fields indexing using LOOKUP.

    Use as args in create_node (using *).
    """
    nodes = [create_node(*row[LOOKUP]) for _, row in df.iterrows()]
    return nodes


def index_series_iterrows_comprehension_ignored_args(df):
    """List comprehension over iterrows.

    Use all fields as kwargs in create_node_ignored_args (using **).
    """
    nodes = [create_node_ignored_args(**row) for _, row in df.iterrows()]
    return nodes


def itertuples(df):
    """Loop over itertuples, convert namedtuple to dict, get fields using tuple_unwrap fn according to LOOKUP.

    Use as kwargs in create_node (using **).
    """
    nodes = []
    for row in df.itertuples(index=False):
        nodes.append(create_node(**tuple_unwrap(row, LOOKUP)))
    return nodes


def itertuples_comprehension(df):
    """List comprehension over itertuples, convert namedtuple to dict, get fields using tuple_unwrap fn according to LOOKUP.

    Use as kwargs in create_node (using **).
    """
    nodes = [create_node(**tuple_unwrap(row, LOOKUP)) for row in df.itertuples(index=False)]
    return nodes


def itertuples2(df):
    """Loop over itertuples, convert namedtuple to a dict, get fields creating a new dict according to LOOKUP.

    Use as kwargs in create_node (using **).
    """
    nodes = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        nodes.append(create_node(**{k: d[k] for k in LOOKUP}))
    return nodes


def itertuples_ignored_args(df):
    """Loop over itertuples, convert namedtuple to dict.

    Use all fields as kwargs in create_node_ignored_args (using **).
    """
    nodes = []
    for row in df.itertuples(index=False):
        nodes.append(create_node_ignored_args(**row._asdict()))
    return nodes


def itertuples_direct_access(df):
    """Loop over itertuples, convert tuple to dict, get fields accessing fields directly.

    Use as kwargs in create_node (using direct assignment).
    """
    nodes = []
    for row in df.itertuples(index=False):
        nodes.append(create_node(b=row.b, a=row.a, f=row.f, e=row.e, d=row.d, c=row.c, g=row.g))
    return nodes


def itertuples_direct_access_comprehension(df):
    """List comprehension over itertuples, get fields accessing them directly.

    Use as kwargs in create_node (using direct assignment).
    """
    nodes = [create_node(b=row.b, a=row.a, f=row.f, e=row.e, d=row.d, c=row.c, g=row.g) for row in df.itertuples(index=False)]
    return nodes


def index_dict_comprehension(df):
    """List comprehension over Convert df to dict, get fields creating another dict according to LOOKUP.

    Use as kwargs in create_node (using **).
    """
    nodes = [create_node(**{k: kwargs[k] for k in LOOKUP}) for kwargs in df.to_dict(orient="records")]
    return nodes


def index_dict_comprehension_ignored_args(df):
    """Convert df to dict, get fields creating another dict according to LOOKUP.

    Use as kwargs in create_node_ignored_args (using **).
    """
    nodes = [create_node_ignored_args(**kwargs) for kwargs in df.to_dict(orient="records")]
    return nodes


def to_records(df):
    """Use to_records on df, get fields indexing using LOOKUP.

    Use as args in create_node (using *).
    """
    nodes = [create_node(*row[LOOKUP]) for row in df.to_records()]
    return nodes


def to_numpy_direct_access(df):
    """Get the names of columns in our dataframe, create indices lookup from name->idx, convert df to numpy and access fields using indices lookup.

    Use as kwargs in create_node (using direct assignment).
    """
    cols = list(df.columns)
    indices = {k: cols.index(k) for k in cols}
    nodes = [
        create_node(
            b=row[indices["b"]],
            a=row[indices["a"]],
            f=row[indices["f"]],
            e=row[indices["e"]],
            d=row[indices["d"]],
            c=row[indices["c"]],
            g=row[indices["g"]],
        )
        for row in df.to_numpy()
    ]
    return nodes


def to_numpy_take(df):
    """Get the names of columns in our dataframe, create indices lookup from name->idx using LOOKUP, convert df to numpy and access fields using the indices lookup using np.take.

    Use as args in create_node (using *).
    """
    cols = list(df.columns)
    indices = [cols.index(k) for k in LOOKUP]
    nodes = [create_node(*np.take(row, indices)) for row in df.to_numpy()]
    return nodes


def zip_comprehension_direct_access(df):
    """List comprehension over zip object, get fields using direct indexing.

    Use as args in create_node (using *).
    """
    nodes = [create_node(*args) for args in zip(df["b"], df["a"], df["f"], df["e"], df["d"], df["c"], df["g"])]
    return nodes


def zip_comprehension_lookup(df):
    """List comprehension over zipped df columns, get fields using * and LOOKUP.

    Use as args in create_node (using *).
    """
    nodes = [create_node(*args) for args in zip(*(df[c] for c in LOOKUP))]
    return nodes


def zip_comprehension_np_values_lookup(df):
    """List comprehension over zipped df columns, get fields using *, LOOKUP, use .values.

    Use as args in create_node (using *).
    """
    nodes = [create_node(*args) for args in zip(*(df[c].values for c in LOOKUP))]
    return nodes


# Roughly ordered from slowest to fastest.
FUNCTIONS = [
    index_series_iterrows,
    index_series_iterrows_comprehension,
    index_df_apply_kwargs,
    index_df_apply,
    index_series_iterrows_comprehension_ignored_args,
    index_dict_iterrows,
    index_dict_iterrows_comprehension,
    index_dict_iterrows_comprehension_ignored_args,
    index_df_apply_kwargs_ignored_args,
    to_records,
    index_dict_apply,
    index_dict_apply_ignored_args,
    index_dict_comprehension,
    index_dict_comprehension_ignored_args,
    to_numpy_take,
    itertuples,
    itertuples_comprehension,
    itertuples2,
    itertuples_ignored_args,
    itertuples_direct_access,
    itertuples_direct_access_comprehension,
    to_numpy_direct_access,
    zip_comprehension_direct_access,
    zip_comprehension_lookup,
    zip_comprehension_np_values_lookup,
]


# Performance test functions.
def setup(n, seed=1):
    """Setup for perfplot, returns a random dataframe with n elements. Uses a seed so every function gets the same input df."""
    random.seed(seed)
    return random_df(n)


def plot():
    """Plot timing using perfplot, saves to perf-clean.png."""
    START_POW_2X = 1
    END_POW_2X = 16  # => 32_768
    SIZES = [2**x for x in range(START_POW_2X, END_POW_2X)]  # Size of our DataFrame: 2, 4, 8, 16, 32, 64, 128, 256, ..., 32768
    LABELS = [str(fn.__name__) for fn in FUNCTIONS]
    out = perfplot.bench(
        setup=lambda n: setup(n),  # Create and return the DataFrame.
        kernels=FUNCTIONS,
        labels=LABELS,
        n_range=SIZES,
        xlabel="N",
        equality_check=lambda x, y: x == y,  # Check that every output is the same.
        show_progress=True,
    )
    out.save(
        "perf-clean-absolute3.png",
        time_unit="us",  # nanoseconds
        logx=True,
        logy=False,
        # relative_to=len(FUNCTIONS) - 1,  # relative to fastest one
    )


def time_low_iteration_count():
    """Small to large DataFrames, low iteration count."""
    START_POW_2X = 1
    END_POW_2X = 16  # => 32_768
    SEEDS = [1, 2, 3, 4, 5]  # To replicate plot() timing, change this to [1].
    SIZES = [2**x for x in range(START_POW_2X, END_POW_2X)]  # Size of our DataFrame: 2, 4, 8, 16, 32, 64, 128, 256, ..., 32768
    print("Starting `time_low_iteration_count` timing test.")
    print(f"Small to large DataFrames, low iteration count. Do {len(SEEDS)} iterations (1 iteration per seed in {SEEDS}) on DataFrame of sizes: {SIZES}.")
    print("")
    fn_mean_time_map = {str(fn.__name__): [] for fn in FUNCTIONS}
    fn_total_time_map = {str(fn.__name__): [] for fn in FUNCTIONS}
    for fn in FUNCTIONS:
        fn_name = f"{fn.__name__}"
        print(f"'{fn_name}':")
        fn_total_time = 0
        for size in SIZES:
            print(f"{size=: <5}:", end=" ")
            times = []
            for seed in SEEDS:
                data = (setup(size, seed=seed),)
                time_taken = perfplot._main._b(data=data, kernel=fn, repeat=1)  # Cludge, but I don't want to setup timeit when I can just use this.
                times.append(time_taken)  # => nanoseconds
            mean = statistics.mean(times)
            fn_mean_time_map[fn_name].append(mean)  # Calculate mean runtime for all the seed iterations, but only for this specific DataFrame size.
            mean_time_per_row = int(mean / size)
            total_time = sum(times)
            fn_total_time += total_time  # Add time that was spent in this DataFrame size for all seeds. This is the total time spent in this one function.
            print(f"{mean//1_000:9,d}μs mean time over {len(SEEDS)} seed iterations, mean time per row: {mean_time_per_row//1_000:3,d}μs, total time taken: {total_time//1_000:11,d}μs.")
        fn_total_time_map[fn_name].append(fn_total_time)
        print(f"Total time taken for '{fn_name}': {fn_total_time//1_000:,d}μs.")
        print("")
    print("Overall timing report:")
    for k, v in sorted(fn_mean_time_map.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
        median = int(statistics.median(v) // 1_000)  # => μs
        mean = statistics.mean(v) // 1_000  # => μs
        mode = statistics.mode(v) // 1_000  # => μs
        mx = max(v) // 1_000  # => μs
        mn = min(v) // 1_000  # => μs
        total = sum(fn_total_time_map[k]) // 1_000  # => μs
        print(f"{k: <48}: {median=:6,d}μs\t{mean=:9,d}μs\tmax={mx:9,d}μs\tmin={mn:5,d}μs\ttotal={total:11,d}μs.")
    print("")
    print("Finished `time_low_iteration_count` timing test.")


def time_high_iteration_count():
    """Time small to medium DataFrames, high iteration count."""
    ITERS = 2**13  # => 8_192
    START_POW_2X = 1
    END_POW_2X = 7  # => 64
    SIZES = [2**x for x in range(START_POW_2X, END_POW_2X)]  # Size of our DataFrame: 2, 4, 8, 16, 32, 64.
    SEED = 1
    print("Starting `time_high_iteration_count` timing test.")
    print(f"Time small to medium DataFrames, high iteration count. Do {ITERS:,d} iterations ({SEED=}) on DataFrame of sizes: {SIZES}.")
    print("")
    fn_mean_time_map = {str(fn.__name__): [] for fn in FUNCTIONS}
    fn_total_time_map = {str(fn.__name__): [] for fn in FUNCTIONS}
    for fn in FUNCTIONS:
        fn_name = f"{fn.__name__}"
        print(f"'{fn_name}':")
        fn_total_time = 0
        for size in SIZES:
            print(f"{size=: <5}:", end=" ")
            times = []
            data = (setup(size, seed=SEED),)
            for _iter in range(ITERS):
                time_taken = perfplot._main._b(data=data, kernel=fn, repeat=1)  # Cludge, but I don't want to setup timeit when I can just use this.
                times.append(time_taken)  # => nanoseconds
            fn_mean_time_map[fn_name].append(statistics.mean(times))  # Calculate mean runtime for all the iterations, but only for this specific DataFrame size.
            mean = statistics.mean(times)
            mean_time_per_row = int(mean / size)
            total_time = sum(times)
            fn_total_time += total_time  # Add time that was spent in this DataFrame size for all iterations. This is the total time spent in this one function.
            print(f"{mean//1_000:9,d}μs mean time over {ITERS:,} iterations, mean time per row: {mean_time_per_row//1_000:3,d}μs, total time taken: {total_time//1_000:11,d}μs.")
        fn_total_time_map[fn_name].append(fn_total_time)
        print(f"Total time taken for '{fn_name}': {fn_total_time//1_000:,d}μs.")
        print("")
    print("Overall timing report:")
    for k, v in sorted(fn_mean_time_map.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
        median = int(statistics.median(v) // 1_000)  # => μs
        mean = statistics.mean(v) // 1_000  # => μs
        mode = statistics.mode(v) // 1_000  # => μs
        mx = max(v) // 1_000  # => μs
        mn = min(v) // 1_000  # => μs
        total = sum(fn_total_time_map[k]) // 1_000  # => μs
        print(f"{k: <48}: {median=:6,d}μs\t{mean=:9,d}μs\tmax={mx:9,d}μs\tmin={mn:5,d}μs\ttotal={total:11,d}μs.")
    print("")
    print("Finished `time_high_iteration_count` timing test.")


if __name__ == "__main__":
    plot()
