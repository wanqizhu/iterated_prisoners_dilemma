import matplotlib
matplotlib.use('Agg')
import axelrod as axl
import os
import numpy as np
import pandas as pd
import tqdm

axl.seed(0)  # Set a seed
#players = [s() for s in axl.demo_strategies]  # Create players
#tournament = axl.Tournament(players)  # Create a tournament
#results = tournament.play()  # Play the tournament
#results.ranked_names

#plot = axl.Plot(results)
#plot.boxplot().savefig('boxplot.png')




parameterized_players = [
        axl.Random(0.1),
        axl.Random(0.3),
        axl.Random(0.7),
        axl.Random(0.9),
        axl.GTFT(0.1),
        axl.GTFT(0.3),
        axl.GTFT(0.7),
        axl.GTFT(0.9),
        axl.MetaWinner(team=[
        axl.EvolvedHMM5, axl.EvolvedLookerUp2_2_2, axl.EvolvedFSM16,
        axl.EvolvedANN5, axl.PSOGambler2_2_2, axl.FoolMeOnce,
        axl.DoubleCrosser, axl.Gradual
    ]),
]

players = [s() for s in axl.strategies] + parameterized_players


"""
A script with utility functions to get the tournament results
"""
from collections import namedtuple
from numpy import median


def label(prefix, results, turns):
    """
    A label used for the various plots
    """
    return "{} - turns: {}, repetitions: {}, strategies: {}. ".format(prefix,
                turns, results.repetitions, results.num_players)

def obtain_assets(results, filename, turns, strategies_name="strategies",
                  tournament_type="std",
                  assets_dir="./assets", lengthplot=False):
    """
    From the results of a tournament: obtain the various plots and the summary
    data set
    Parameters
    ----------
        results: axelrod.resultset instance
        strategies_name: string, eg: "ordinary_strategies"
        tournament_type: string, eg: "std"
        assets_dir: string [default: "./assets"]
        lengthplot: boolean [default: False], whether or not to plot the length
        of the matches (only needed for the probabilistic ending matches)
    """

    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    total = 6 + int(lengthplot)

    pbar = tqdm.tqdm(total=total, desc="Obtaining plots")

    file_path_root = "{}/{}_{}".format(assets_dir, strategies_name,
                                       tournament_type)
    plot = axl.Plot(results)

    f = plot.boxplot(title=label("Payoff", results, turns))
    f.savefig("{}_boxplot.png".format(file_path_root))
    f.savefig("{}_boxplot.svg".format(file_path_root))
    pbar.update()

    f = plot.payoff(title=label("Payoff", results, turns))
    f.savefig("{}_payoff.png".format(file_path_root))
    f.savefig("{}_payoff.svg".format(file_path_root))
    pbar.update()

    f = plot.winplot(title=label("Wins", results, turns))
    f.savefig("{}_winplot.png".format(file_path_root))
    f.savefig("{}_winplot.svg".format(file_path_root))
    pbar.update()

    f = plot.sdvplot(title=label("Payoff differences", results, turns))
    f.savefig("{}_sdvplot.png".format(file_path_root))
    f.savefig("{}_sdvplot.svg".format(file_path_root))
    pbar.update()

    f = plot.pdplot(title=label("Payoff differences", results, turns))
    f.savefig("{}_pdplot.png".format(file_path_root))
    f.savefig("{}_pdplot.svg".format(file_path_root))
    pbar.update()

    eco = axl.Ecosystem(results)
    eco.reproduce(1000)
    f = plot.stackplot(eco, title=label("Eco", results, turns))
    f.savefig("{}_reproduce.png".format(file_path_root))
    f.savefig("{}_reproduce.svg".format(file_path_root))
    pbar.update()

    if lengthplot is True:
        f = plot.lengthplot(title=label("Length of matches", results, turns))
        f.savefig("{}_lengthplot.png".format(file_path_root))
        f.savefig("{}_lengthplot.svg".format(file_path_root))
        pbar.update()
        
    return plot


def run_tournament(players, turns, repetitions, filename, tournament_type="std", assets_dir="./assets", match_attributes=None):
    axl.seed(seed)  # Setting a seed

    tournament = axl.Tournament(players, turns=turns, repetitions=repetitions, match_attributes=None)

    results = tournament.play(filename=filename, processes=processes)
    obtain_assets(results, filename, turns, "strategies", tournament_type, assets_dir)
    results.write_summary(assets_dir + '/' + tournament_type + '_summary.csv')
    return results


turns = 3
repetitions = 2

processes = 0
seed = 1
filename = "data/strategies_std_interactions.csv"

results = run_tournament(players, turns, repetitions, filename, "std", "./assets", None)

#results_summary = results.summarise()

results_summary = pd.read_csv('assets/std_summary.csv', index_col='Rank')

bad_strategies = set(results_summary[results_summary['Initial_C_rate'] != 1]['Name'])
good_players = [p for p in players if p.name not in bad_strategies]

turns = 50
repetitions = 5

filename = "data/tour2_strategies_std_interactions.csv"

tour2_results = run_tournament(good_players, turns, repetitions, filename, "tour2", "./tour2_assets", match_attributes={"length": float('inf')})


