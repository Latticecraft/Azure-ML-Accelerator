import os
import argparse
import pandas as pd
import glob

from pathlib import Path


def main(args):
    df_runinfo = compactify(args.runinfo)
    df_trainlog = compactify(args.trainlog)

    # ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)

    # save data to outputs
    df_runinfo.to_csv('outputs/runinfo-history.csv', index=False)
    df_runinfo.to_csv((Path(args.runinfo) / 'history.csv'), index=False)
    df_trainlog.to_csv('outputs/trainlog-history.csv', index=False)
    df_trainlog.to_csv((Path(args.trainlog) / 'history.csv'), index=False)


def compactify(path):
    try:
        df_all = pd.read_csv((Path(path) / 'history.csv'))
    except:
        df_all = pd.DataFrame()

    # list files in folder
    deltas = glob.glob(path+'/*')
    for d in deltas:
        print('adding {}'.format(d))
        df_delta = pd.read_csv((Path(path) / d))
        df_all = pd.concat([df_all, df_delta], ignore_index=True)

    # delete deltas
    for d in deltas:
        os.remove((Path(path) / d))

    return df_all


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--runinfo', type=str, default='data/runinfo')
    parser.add_argument('--trainlog', type=str, default='data/trainlog')

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == '__main__':
    # parse args
    args = parse_args()

    # run main function
    main(args)