import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser


DEFAULT_PATH = os.path.join(os.path.curdir, "MackeyGlass.csv")

MACKEY_GLASS_N   = 10000
MACKEY_GLASS_B   = 0.1
MACKEY_GLASS_C   = 0.2
MACKEY_GLASS_TAU = 25

def generate_mackey_glass(
    n_points: int = MACKEY_GLASS_N,
    b: float = MACKEY_GLASS_B,
    c: float = MACKEY_GLASS_C,
    tau: int = MACKEY_GLASS_TAU,
    plot: bool = False,
    path: str = ""
) -> pd.DataFrame:
    
    # Initial conditions
    y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
        1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]
    
    # Generate Data
    for n in range(17,n_points+99):
        y.append(y[n] - b*y[n] + c*y[n-tau]/(1+y[n-tau]**10))
    y = np.array(y[100:])
    df = pd.DataFrame({"Total": y})
    
    # Save to csv
    if path:
        df.to_csv(path)
        print(f"New Mackey-Glass dataset saved in {path}")

    # Plot generated
    if plot:
        plt.plot(y[:1000])
        plt.ylim(0, 2)
        plt.show()
    
    return df


if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--n', default=MACKEY_GLASS_N, type=int)
    parser.add_argument('--b', default=MACKEY_GLASS_B, type=float)
    parser.add_argument('--c', default=MACKEY_GLASS_C, type=float)
    parser.add_argument('--tau', default=MACKEY_GLASS_TAU, type=int)
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--path', default=DEFAULT_PATH, type=str)

    # Generate Mackey-Glass dataset
    args = parser.parse_args()
    generate_mackey_glass(args.n, args.b, args.c, args.tau, args.plot, args.path)
