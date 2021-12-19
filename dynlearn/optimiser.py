"""Optimisers and targets.

Provides an interface to allow comparison of active learning of Gaussian process dynamical systems
with other optimisers such as Bayesian optimisation and Powell's method."""

import logging
import itertools
import operator as op
import numpy as np
import pandas as pd
import altair as alt
import tensorflow as tf

logger = logging.getLogger(__name__)


def min_so_far(iterable):
    """Cumulative minimum."""
    smallest = None
    for x in iterable:
        if smallest is None or x < smallest:
            smallest = x
        yield smallest


class OptimisationTarget:
    """A function object that can be the target of an optimiser.

    Records function evaluations and associated simulations.
    """

    def __init__(self, sim, loss_fn, knots):
        self.sim = sim
        self.loss_fn = loss_fn
        self.knots = knots
        self.knot_values = []
        self._history = []
        self.losses = []

    @property
    def history(self):
        return np.asarray(self._history)

    @property
    def recent_tracks(self):
        return self._history[-1]

    @staticmethod
    def _squeeze_knot_values(knot_values):
        """Sometimes knot values are supplied with extra leading dimensions of size 1, so squeeze these out."""
        while knot_values.ndim > 1:
            knot_values = np.squeeze(knot_values, axis=0)
        return knot_values

    def __call__(self, knot_values):
        """The function to minimise."""

        # Convert knot values to u tracks
        knot_values = OptimisationTarget._squeeze_knot_values(knot_values)
        self.knot_values.append(np.array(knot_values))
        u_tracks = self.sim.u_tracks_from_knots(knots=self.knots, knot_values=knot_values)

        # Simulate
        tracks = self.sim.tracks_for_u(u_tracks.T)
        self._history.append(tracks)  # Remember simulation history

        # Calculate loss
        loss = self.loss_from_tracks(tracks)
        self.losses.append(loss)
        return loss

    def loss_from_tracks(self, tracks):
        # Calculate loss as function of simulation output
        with tf.Session().as_default():
            return self.loss_fn.mean_loss(tracks.T[np.newaxis], tracks[:self.loss_fn.u_dim].T).eval()


def array_as_series(a, indexes, name='value'):
    if a.ndim != len(indexes):
        raise ValueError(f'Must have an index for each dimension: {a.ndim} != {len(indexes)}')
    indexes = [(index_name, np.arange(a.shape[d]) if index is None else index)
               for d, (index_name, index)
               in enumerate(indexes)]
    for d, (_, index) in enumerate(indexes):
        if a.shape[d] != len(index):
            raise ValueError(f'Index for dimension {d} is wrong length: {a.shape[d]} != {len(index)}')
    index = pd.MultiIndex.from_product(list(map(op.itemgetter(1), indexes)), names=list(map(op.itemgetter(0), indexes)))
    return pd.Series(index=index, data=a.flatten(), name=name)


def epochs_to_show(n_epochs, max_epochs):
    return np.linspace(0, n_epochs - 1, num=max_epochs, dtype=int)


def chart_history(sim, history, max_epochs=12, samples=None):
    """Create an altair chart of the history of the optimisation."""
    # This may be Nanog specific. TODO: check how to generalise to other demos
    control_vars = ['U (scaled)']
    n_epochs = history.shape[0]
    species = list(itertools.chain(control_vars, sim.output_vars))
    tracks_df = array_as_series(history, (('epoch', None),
                                          ('species', species),
                                          ('time_step', None))).reset_index()
    tracks_df['is_sample'] = False
    tracks_df['sample'] = -1
    #
    # Combine with samples if we have been given them
    if samples is not None:
        samples_df = array_as_series(samples, (('epoch', None),
                                               ('sample', None),
                                               ('time_step', None),
                                               ('species', species))).reset_index()
        samples_df['is_sample'] = True
        tracks_df = pd.concat([tracks_df, samples_df], ignore_index=True)
    tracks_df['control'] = tracks_df['species'].isin(control_vars)
    tracks_df = tracks_df[tracks_df['epoch'].isin(epochs_to_show(n_epochs, max_epochs))]
    opacity = alt.condition(alt.datum.is_sample, alt.value(.3), alt.value(1.))
    chart = (
        alt.Chart(tracks_df)
        .mark_line()
        .encode(x='time_step:O',
                y='value:Q',
                color='species:N',
                strokeDash='control:N',
                detail='sample',
                opacity=opacity,
                facet=alt.Facet('epoch:O', columns=3)))
    return chart


def optimise_random(sim, loss_fn, knots, knot_values, args):
    """Use random sampling to control system."""

    # Define target function of optimiser
    target = OptimisationTarget(sim, loss_fn, knots)

    best, best_loss = None, np.inf
    losses = []
    for epoch in range(args.num_epochs):
        loss = target(knot_values)
        losses.append(losses)
        if loss < best_loss:
            best = knot_values
            best_loss = loss
        knot_values = np.random.uniform(low=0, high=args.u_max, size=len(knots))

    logger.info('Minimised loss to {:.2f} using {} function evaluations'.format(best_loss, args.num_epochs))
    logger.info('Optimal inputs: {}'.format(np.round(best, 2)))

    return dict(target=target, history=target.history, best_u=best, losses=losses, args=args)


def optimise_powell(sim, loss_fn, knots, knot_values, args):
    """Use scipy Powell optimiser to control system."""
    import scipy

    # Define target function of optimiser
    target = OptimisationTarget(sim, loss_fn, knots)

    # Optimise with scipy's Powell method
    res = scipy.optimize.minimize(target, x0=knot_values, method='Powell', options=dict(maxfev=args.num_epochs),
                                  bounds=[(0, args.u_max)] * len(knot_values))
    assert res.nfev >= args.num_epochs
    logger.info(res.message)
    logger.info('Minimised loss to {:.2f} in {} iterations using {} function evaluations'.format(
        res.fun, res.nit, res.nfev))
    logger.info('Optimal inputs: {}'.format(np.round(res.x, 2)))

    return dict(target=target, history=target.history, best_u=res.x, losses=target.losses, args=args)


# TODO: check whether we should use knot_values like other optimisers
def optimise_bayesian(sim, loss_fn, knots, knot_values, args):
    """Use Bayesian optimisation to control system."""
    import GPy
    from GPyOpt.methods import BayesianOptimization

    # Define target function of optimiser
    target = OptimisationTarget(sim, loss_fn, knots)

    # Domain over which we will optimise
    domain = [{'name': 'knot_values', 'type': 'continuous', 'domain': (0, args.u_max), 'dimensionality': len(knots)}]

    # Define Gaussian process to model loss and optimiser
    kernel = GPy.kern.Matern52(input_dim=len(knots), variance=2**2, lengthscale=args.u_max / 3)
    optimiser = BayesianOptimization(f=target, X=knot_values[np.newaxis], domain=domain, kernel=kernel,
                                     noise_var=.05**2, maximize=False)

    # Optimise
    optimiser.run_optimization(max_iter=args.num_epochs)
    logger.info('Optimised loss to {:.2f} using {} function evaluations'.format(
        optimiser.fx_opt, len(optimiser.get_evaluations()[1])))
    assert len(optimiser.get_evaluations()[1]) == len(target.history)
    logger.info('Optimal inputs: {}'.format(np.round(optimiser.x_opt, 2)))

    return dict(target=target, history=target.history, best_u=optimiser.x_opt, losses=target.losses, args=args)


def optimise_active(sim, loss_fn, gp, knots, knot_values, args):
    """Optimise using active learning of Gaussian process dynamical system."""

    from dynlearn import learn as lf

    # Define target function of optimiser
    target = OptimisationTarget(sim, loss_fn, knots)

    # TODO: check x0 used similarly in other optimisers
    x0 = np.zeros(len(sim.output_vars))
    epoch_results = lf.search_u(target=target,
                                gp=gp,
                                knot_values=knot_values,
                                x0=x0,
                                u_max_limit=args.u_max,
                                n_epochs=args.num_epochs,
                                n_samples=args.num_samples,
                                is_nonnegative=args.is_nonnegative,
                                predict_random=args.predict_random,
                                is_epsilon=args.is_epsilon,
                                is_diff=args.gp_diff)

    # Wrangle results into same format as other optimisers and return
    return dict(epoch_results=epoch_results,
                target=target,
                history=target.history,
                losses=target.losses,
                best_u=epoch_results[-1]['u'],
                args=args)


def optimise(sim, loss_fn, gp, knots, knot_values, args):
    """Dispatch optimisation to an optimiser as configured in `args`."""
    # TODO: let each optimiser return best knot values
    # TODO: check knot values are handled consistently across optimisers,
    # that is each optimiser uses the knot values to initialise with
    if 'random' == args.optimiser:
        return optimise_random(sim, loss_fn, knots, knot_values, args)

    if 'active' == args.optimiser:
        return optimise_active(sim, loss_fn, gp, knots, knot_values, args)

    elif 'Bayesian' == args.optimiser:
        return optimise_bayesian(sim, loss_fn, knots, knot_values, args)

    elif 'Powell' == args.optimiser:
        return optimise_powell(sim, loss_fn, knots, knot_values, args)

    else:
        raise ValueError(f'Unknown optimiser: {args.optimiser}')
