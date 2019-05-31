import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import argparse
from model import KaggleModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trainFile", help="The training file to be used")
    parser.add_argument("testFile", help="The testing file to be used")
    parser.add_argument("actions", nargs='*', help="The action to be performed by the model")
    args= parser.parse_args()
    model = KaggleModel(args.trainFile, args.testFile)
    print(dir(model))
    for act in args.actions:
        f = getattr(model, act)
        f()
