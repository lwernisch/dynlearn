"""
The module provides functionality to optimise the input to an unknown dynamical
system (but with known or estimated dimension) to achieve a certain production
level for a target species at a specified time point.

"""

import numpy as np
import argparse
from collections import namedtuple
from dynlearn import simulation as sf, learn as lf


def setup(args):
    if 'nanog50' == args.demo:
        return nanog_target(args=args)
    if 'ffl780' == args.demo:
        return ffl_target(args=args)
    else:
        raise ValueError('Unknown demo: {}'.format(args.demo))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', default='nanog50', help='Which demo to use')
    parser.add_argument('-S', '--num-samples', type=int, default=10,
                        help='Number of samples to draw from GP dynamical model')
    parser.add_argument('-T', '--num-times', type=int, default=20, help='Number of simulation steps')
    parser.add_argument('-E', '--num-epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--u-max', type=float, default=1000, help='Maximum value for control inputs')
    parser.add_argument('--seed', type=int, default=123456, help='RNG seed')
    parser.add_argument('--device', default='', help='Device to run on')
    parser.add_argument('--optimiser', default='active', choices=('active', 'Bayesian', 'Powell', 'random'),
                        help='Optimiser to run')
    return parser


def tag_from_args(args):
    "Construct a string that represents the arguments."
    return f'{args.demo}-{args.optimiser}-{args.num_samples}-{args.num_times}-{args.num_epochs}-{args.seed}-{args.u_max}'


_ArgsDuckType = namedtuple('_ArgsDuckType',
                           ['demo', 'optimiser', 'num_samples', 'num_times', 'num_epochs', 'seed', 'u_max'])


def args_from_tag(tag):
    "Reconstruct the arguments from a tag."
    split = tag.split('-')
    return _ArgsDuckType(demo=split[0],
                         optimiser=split[1],
                         num_samples=int(split[2]),
                         num_times=int(split[3]),
                         num_epochs=int(split[4]),
                         seed=int(split[5]),
                         u_max=float(split[6]))


def ffl_target(args):
    """Optimise input so that the level of ``x3`` goes to 780.

    The simulator is a feed-forward loop.
    """
    # I think real_time is the number of time units to simulate
    sim = sf.FeedForwardOrCSimulation(n_times=args.num_times, real_time=20)

    # Choose a loss function that wants X2 to have expression level of 780
    loss = lf.RegularisedEndLoss(
        target=780.0, target_ind=sim.output_vars.index('X2'),
        u_dim=1, time_ind=sim.n_times - 1, reg_weights=0.0)

    # Use a squared exponential covariance function with given length scales
    # and variances
    gp = lf.FixedGaussGP(
        lengthscales=np.array([100.0 ** 2, 100.0 ** 2, 100.0 ** 2, 100.0 ** 2]),
        variance=5 ** 2, likelihood_variance=2 ** 2)

    # Choose the knots at which we can control the input and the initial values
    knots = np.array([0, 5, 10])
    knot_values = np.array([300.0, 200.0, 100.0])

    # Plotting arguments
    plot_args = {'ylim': (0, 900)}

    return sim, loss, gp, knots, knot_values, plot_args


def nanog_target(args):
    """Optimise input so that a target level of NANOG is produced in the Biomodel
    `Chickarmane2006 - Stem cell switch reversible
    <https://www.ebi.ac.uk/biomodels/BIOMD0000000203>`_
    The model is simulated using a ODE solver of the
    `Tellurium <http://tellurium.analogmachine.org/>`_
    package for biomolecular models.

    Args:
        args: configuration

    To optimise the output a Gaussian process state space model (GPSSM) is
    constructed from an initial and ``n_epochs`` follow-up experiments. All
    species levels of the system at a particular simulation step are input to
    the GP, and the increase or decrease in the next simulation step is the
    output, ie, there is one GP for each species (assumed to be independent
    conditional on the common input). The settings for the Gaussian process
    gp parameters (lengthscales for the inputs, variance, and error
    variance for the output squared exponential gp) are chosen manually
    to fit the range of the variables.

    The aim is to achieve a level of 50 for NANOG by simulation step 20 (real
    time 10). Input is only allowed at steps 0, 5, and 10. The input is
    limited to [0.0,1000.0].

    The optimisation takes a few minutes on a typical workstation. However,
    depending on random settings it might take more or fewer epochs to find
    an input that induces the desired level of NANOG.

    Returns:
        Successive optimisation results are stored in a results file
        than can be loaded and displayed by running ``scripts/demo-nanog-plots``

    Call for example under Unix by::

     python3 scripts/demo-nanog
     python3 dynlearn/demo-nanog-plots
     display dynlearn/results/Nanog_target_50.png

    """

    # I think real_time is the number of time units to simulate
    sim = sf.StemCellSwitch(n_times=args.num_times, real_time=10)

    # Choose a loss function that wants NANOG to have expression level of 50
    loss = lf.RegularisedEndLoss(
        target=50.0, target_ind=sim.output_vars.index('NANOG'),
        u_dim=1, time_ind=sim.n_times - 1, reg_weights=0.0)

    # Use a squared exponential covariance function with given length scales
    # and variances
    gp = lf.FixedGaussGP(
        lengthscales=np.array([100.0 ** 2, 100.0 ** 2, 100.0 ** 2, 20 ** 2,
                               100.0 ** 2, 100.0 ** 2]),
        variance=5 ** 2, likelihood_variance=2 ** 2)

    # Choose the knots at which we can control the input and the initial values
    knots = np.array([0, 5, 10])
    knot_values = np.array([200.0, 150.0, 100.0])

    # Plotting arguments
    plot_args = {'ylim': (0, 80)}

    return sim, loss, gp, knots, knot_values, plot_args
