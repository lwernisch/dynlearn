"""Core module of package optimising the forcing input to the dynamical
system to achieve a target level of species

See :mod:`dynlearn.demo` for an example how to use this module
"""
import tensorflow as tf
import numpy as np
import time
import logging
from dynlearn import gp_kernels as gpf
from dynlearn import simulation

logger = logging.getLogger(__name__)


# -------    Loss classes

class Loss:
    """Generic loss class"""
    def mean_loss(self, rtracks_lst, u):
        pass

    def mean_target(self, rtracks_lst_eval):
        pass


class RegularisedEndLoss(Loss):
    """Defines a loss function suitable for optimising the forcing input,

    Args:
        target (float): value or 9999.0 for maximising, -9999.0 for minimising
        target_ind (int): variable index of target variable (without u)
        u_dim (int): dimension of forcing input part of tracks vector
        time_ind: step index at which to compute loss
        reg_weights: L1 regularisation terms
    """

    def __init__(self, target, target_ind, u_dim, time_ind, reg_weights=0.5):
        self.target = target
        self.target_ind = target_ind
        self.u_dim = u_dim
        self.time_ind = time_ind
        self.reg_weights = reg_weights

    def mean_loss(self, rtracks_lst, u):
        total_loss = 0.0
        # sumsq = 0.0
        target_u_ind = self.u_dim + self.target_ind
        for rtracks in rtracks_lst:
            if self.target == 9999.0:
                loss = - tf.square(rtracks[self.time_ind, target_u_ind])
            elif self.target == -9999.0:
                loss = tf.square(rtracks[self.time_ind, target_u_ind])
            else:
                loss = tf.square(rtracks[self.time_ind, target_u_ind] - self.target)
            total_loss += loss
            # sumsq += loss**2
        mean_loss = total_loss / len(rtracks_lst) + self.reg_weights * tf.reduce_sum(tf.abs(u))
        # sd_loss = tf.sqrt(sumsq/n_lst - mean_loss**2)
        return mean_loss  # - 2.0*sd_loss

    def mean_target(self, rtracks_lst_eval):
        total_loss = 0
        for rtracks in rtracks_lst_eval:
            total_loss += rtracks[self.time_ind, self.u_dim + self.target_ind]
        return total_loss / len(rtracks_lst_eval)


# ------- Kernel classes

class FixedGaussGP:
    """A simple GP class with the usual squared exponential kernel
    parameters
    """

    def __init__(self, lengthscales, variance, likelihood_variance):
        self.lengthscales = lengthscales
        self.variance = variance
        self.likelihood_variance = likelihood_variance

    def kernel_for_u(self, u_tracks, sim, k=None, is_diff=True):
        """Runs the simulation *sim* for the forcing inputs *u_tracks* and
        adds the resulting input-output relationship to the GP *k*.

        Args:
            u_tracks (ndarray): the new forcing inputs
            sim (simulation.Simulation): the experiment simulator
            k (gp_kernels.Kernel): represents the GP, if None newly created
            is_diff (bool): differences are modelled

        Returns:
            A tuple with (*t* is number steps, *d* state dim including
            forcing input, *m* is species dim only)

                - **k** (*gp_kernels.Kernel*): the created or updated GP
                - **X_span** (*(t-1,d) np.array*): new GP inputs
                - **Y_span** (*(t-1,m) np.array*): new GP output
        """
        tracks, u_dim, X_span, Y_span = simulation.simulate(sim, u_tracks)
        # Calculate the output of the GP
        if is_diff:
            # The difference between consecutive time points
            Y_gp = tracks[u_dim:, 1:].T - tracks[u_dim:, :-1].T
        else:
            Y_gp = Y_span
        #
        # Do we have a kernel?
        if k is None:
            #
            # Choose a default kernel as none given
            k = gpf.Kernel(lengthscales=self.lengthscales,
                           variance=self.variance,
                           likelihood_variance=self.likelihood_variance,
                           x=X_span)
            k.set_y(Y_gp)
        else:
            #
            # add to existing kernel
            k.add_x(X_span)
            k.add_y(Y_gp)

        return (k, X_span, Y_span)


# -------------  Core estimation functions

def make_u_col_tf(u_col, trainable_inds, u_type, u_max_limit=None,
                  u_min_limit=0.0):
    """
    Returns the TF version of forcing input *u* with trainable structure
    according to either a 'peak' or 'step' version of the input

    Args:
        u_col (ndarray): forcing *d*-dim inputs to dynamical system
        trainable_inds (List[int]): simulation steps where input can be optimised

    Returns:
        *(t,d)* tf.Tensor containing the trainable TF variables
        corresponding to input
    """

    u_trainable = np.full(len(u_col), False)
    u_trainable[trainable_inds] = True
    u_lst = []
    if u_type == "peak":
        for i in range(len(u_trainable)):
            tf_var = tf.maximum(tf.Variable(u_col[i, ], trainable=u_trainable[i],
                                            dtype="float64"), u_min_limit)
            if u_max_limit is not None:
                tf_var = tf.minimum(tf_var, u_max_limit)
            u_lst.append(tf_var)
    elif u_type == "step":
        j = 0  # u's before the first knot are all fixed
        while not u_trainable[j]:
            tf_var = tf.maximum(
                tf.Variable(u_col[j, ], trainable=False, dtype="float64"), 0.0)
            if u_max_limit is not None:
                tf_var = tf.minimum(tf_var, u_max_limit)
            u_lst.append(tf_var)
            j += 1
        for i in range(j, len(u_trainable)):
            if u_trainable[i]:  # know for i = j is definitely trainable
                tf_current = tf.maximum(
                    tf.Variable(u_col[i, ], trainable=True, dtype="float64"),
                    0.0)
                if u_max_limit is not None:
                    tf_current = tf.minimum(tf_current, u_max_limit)
            u_lst.append(tf_current)  # u's are copies of current knot toleft
    else:
        raise ValueError("unknown u_type {}".format(u_type))
    return tf.stack(u_lst, 0)


def search_u(sim, loss, gp, knots, knot_values, x0, u_max_limit=None,
             n_epochs=6, n_samples=10):
    """Main forcing input optimisation function using a GP to approximate
    a dynamical system given by a simulator. In ``n_epochs`` rounds of
    suggesting a forcing input followed by an experiment and improvement of
    the GP the system is explored. In order to capture the GP uncertainty,
    ``n_samples`` realisations of the recursive GP simulation are created.
    The loss function can make use of these alternative pathways, eg by
    only considering the mean or taking the variance into account.

    Args:
        sim (simulation.Simulation): the simulator for experiments
        loss (Loss): target loss
        gp (FixedGaussGP): GP for estimating dynamical system
        knots (List[int]): simulation steps where input can be optimised
        knot_values (List[float]): starting values for forcing input
        x0 (List[float]): starting values for nonforced species
        u_max_limit (float): limit on maximum for forcing input
        n_epochs (int): number of experiments which can be performed
        n_samples (int): number of random realisations from the GP recursion

    Returns:
        A list with one item per epoch. The item contains

            - the output of `FixedGaussGP.kernel_for_u` for the experiment
            - `X_span`
            - `Y_span`
            - **u_col** (*np.array*): the forcing inputs for the experiment

    """
    #
    # Calculate the values of u across all time points from the knots and their values
    u_col = sim.u_tracks_from_knots(sim.n_times, knots, knot_values).T
    #
    # Run the simulation and update the GP with new data
    k, X_span, Y_span = gp.kernel_for_u(u_tracks=u_col.T, sim=sim)
    n_steps = u_col.shape[0] - 1
    result_lst = [[k, X_span, Y_span, u_col]]

    with tf.Session() as sess:
        for epoch in range(n_epochs):
            logger.info("Epoch {}: start with u_tracks {}".format(epoch, np.round(u_col.T[:, knots], 2)))
            logger.info("Epoch {}: current sim achieves {:.2f}".format(epoch, Y_span[n_steps - 1, loss.target_ind]))

            #
            # Construct TF variables for the forcing inputs
            u_col_tf = make_u_col_tf(u_col=u_col, trainable_inds=knots,
                                     u_type=sim.u_type,
                                     u_max_limit=u_max_limit)
            #
            # TODO: Not sure we need/should reinitialise every epoch. Is there a reason
            # this is here?
            sess.run(tf.global_variables_initializer())

            #
            # Construct a TensorFlow computation graph to
            # run n_samples through the system defined by the GP
            rtracks_lst = []
            for i in range(n_samples):
                rtracks_lst.append(k.tf_recursive(u_col_tf=u_col_tf, x0=x0,
                                                  is_epsilon=True,
                                                  is_random=True,
                                                  is_nonnegative=True))

            #
            # Construct a TensorFlow computation graph to
            # average loss over samples
            mean_loss = loss.mean_loss(rtracks_lst, u_col_tf)

            #
            # Minimise the loss using scipy optimizer
            time0 = time.time()
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(mean_loss)  # ,method="CG")
            optimizer.minimize(sess)

            #
            # Evaluate the average loss tensor
            mean_loss_eval, u_col = sess.run([mean_loss, u_col_tf])
            logger.info("Epoch {}: loss {:.2f} with u_col {} in time {:.1f}s".format(
                epoch, mean_loss_eval, np.round(u_col.T[:, ], 2), np.round(time.time() - time0, 2)))

            #
            # Evaluate the system
            rtracks_lst_eval = sess.run(rtracks_lst)
            logger.info("Epoch {}: mean target {:.2f} of target {:.2f}".format(
                epoch, loss.mean_target(rtracks_lst_eval), loss.target))

            u_sim = sim.u_tracks_from_knots(sim.n_times, knots, u_col.T[0, knots]).T
            k, X_span, Y_span = gp.kernel_for_u(u_tracks=u_sim.T, sim=sim, k=k)

            result_lst.append([k, X_span, Y_span, u_col])

            logger.info("Epoch {}: end with u_col {}".format(epoch, np.round(u_col.T[:, knots], 2)))
            logger.info("Epoch {}: sim achieves {:.2f}".format(epoch, Y_span[n_steps - 1, loss.target_ind]))

        # end for loop
    # end with

    return result_lst
