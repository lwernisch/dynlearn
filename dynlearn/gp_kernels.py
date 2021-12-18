"""Basic Gaussian process functionality for squared exponential kernel

The :class:`Kernel` class implements a basic GP methods for numpy inputs, and
parallel versions for `Tensorflow <https://www.tensorflow.org/>`_. This is
a very basic GP version not providing any optimisation of hyperparameters.

Example:
    Optimise a latent test variable to fit target distribution::

        # 2d input, 5 sample points
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                      [7.0, 8.0], [9.0, 10.0]])
        # corresponding 2-multi output:
        y = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6],
                      [7.7, 8.8], [9.9, 10.1]])

        k = Kernel(lengthscales=1.1 * np.ones(x.shape[1]),
                   variance=1.2 ** 2,
                   likelihood_variance=0.001 ** 2,
                   x=x)
        k.set_y(y)

        # set up latent test input for prediction:
        z = tf.Variable([[0.5, 2.5], [3.5, 4.5]], dtype='float64')
        sess.run(tf.global_variables_initializer())

        m_target = np.array([[1.1, 2.2], [3.3, 4.4]], dtype='float64')
        m_tf, v_tf = k.tf_predict(z)
        sess.run(m_tf) # mean prediction over z far from m_target

        # loss(z) is log prob(m_target | z, k):
        mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(m_tf, v_tf)
        loss = -tf.reduce_sum(mvn.log_prob(m_target))

        # optimise latent z
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss)
        optimizer.minimize(sess)

        # latent test input should be close to true x input:
        z_eval = sess.run(z)
        assert np.sum(np.abs(z_eval - k.x[0:2, :])) <  0.01

"""

import tensorflow as tf
import numpy as np


class Kernel:
    """GP methods for squared exponential kernel. After definition using
    lengthscales

    Args:
        lengthscales: lengthscales of input
        variance: amount of variation
        likelihood_variance: error variance
        x: input
    """
    def __init__(self, lengthscales, variance, likelihood_variance, x=None):
        self.lengthscales = np.float64(lengthscales)
        self.variance = np.float64(variance)
        self.likelihood_variance = np.float64(likelihood_variance)
        if x is not None:
            self.set_x(x)

    def compute_x_stuff(self):
        self.Kxx = self.np_Kxx(is_epsilon=True)
        self.Kinv = np.float64(np.linalg.inv(self.Kxx))
        self.input_dim = self.x.shape[1]

    def set_x(self, x):
        """Set the input values for GP

        Args:
            x ((n,d) np.array): n inputs of dim d
        """
        self.x = np.float64(x.copy())
        self.compute_x_stuff()

    def add_x(self, x):
        """Add new samples to the current input values"""
        self.x = np.vstack([self.x, np.float64(x.copy())])
        self.compute_x_stuff()

    def compute_y_stuff(self):
        self.Kinv_y = np.float64(np.linalg.solve(self.Kxx, self.y))
        self.output_dim = self.y.shape[1]

    def set_y(self, y):
        """Set the output (target) values

        Args:
            y ((n,k) np.array): k independent multioutputs for n inputs
        """
        self.y = np.float64(y.copy())
        self.compute_y_stuff()

    def add_y(self, y):
        """Add new samples to the current output values"""
        self.y = np.vstack([self.y, y.copy()])
        self.compute_y_stuff()

    def np_Kvw(self, v, w, is_epsilon=False):
        v1 = np.expand_dims(v, 1)  # (m,1,input_dim), along rows
        w1 = np.expand_dims(w, 0)  # (1,n,input_dim), along cols
        s1 = np.expand_dims(np.expand_dims(self.lengthscales, 0),
                            0)  # (1,1,input_dim)
        sq_sum = np.sum(np.square(v1 - w1) / s1, axis=2)  # sum along input_dim
        Kvw = self.variance * np.exp(-sq_sum)
        if is_epsilon:  # and Kvw.shape[0] == Kvw.shape[1]:
            Kvw = Kvw + self.likelihood_variance * np.eye(Kvw.shape[0])
        return Kvw

    def np_Kxx(self, is_epsilon=False):
        return self.np_Kvw(self.x, self.x, is_epsilon)

    def np_Kzx(self, z, is_epsilon=False):
        return self.np_Kvw(z, self.x, is_epsilon)

    def np_Kzz(self, z, is_epsilon=False):
        return self.np_Kvw(z, z, is_epsilon)

    def np_predict(self, z, is_epsilon=False, is_cov=True):
        """Predict output for new inputs

        Args:
            z ((m,d) np.array): *m* input points of dim *d* (same as **x**)
            is_epsilon (boolean): add measurement error
            is_cov (boolean): return covariance matrix

        Returns:
            tuple containing

                - **mean** (*(m,k) np.array*): *m* mean predictions over **z** for
                    *k* outputs
                - **var** (*(m,m) np.array*): covariance of **z** inputs
        """
        Kzx = self.np_Kzx(z, False)
        mean = np.matmul(Kzx, self.Kinv_y)
        var = None
        if is_cov:
            Kzz = self.np_Kzz(z, is_epsilon)
            var = Kzz - np.matmul(np.matmul(Kzx, self.Kinv), np.transpose(Kzx))
        return (mean, var)

    def tf_Kvw(self, v, w, is_epsilon=False):
        v1 = tf.expand_dims(v, 1)  # (n,1,input_dim), along rows
        w1 = tf.expand_dims(w, 0)  # (1,n,input_dim), along cols
        s1 = tf.expand_dims(tf.expand_dims(self.lengthscales, 0),
                            0)  # (1,1,input_dim)
        sq_sum = tf.reduce_sum(tf.square(v1 - w1) / s1,
                               axis=2)  # sum along input_dim
        Kvw = self.variance * tf.exp(-sq_sum)
        if is_epsilon:  # and tf.shape(Kvw)[0] == tf.shape(Kvw)[1]:
            Kvw = Kvw + self.likelihood_variance * tf.eye(tf.shape(Kvw)[0],
                                                          dtype='float64')
        return Kvw

    def tf_Kzx(self, z, is_epsilon=False):
        return self.tf_Kvw(z, self.x, is_epsilon)

    def tf_Kzz(self, z, is_epsilon=False):
        return self.tf_Kvw(z, z, is_epsilon)

    def tf_predict(self, z, is_epsilon=False, is_cov=True):
        """Predict output for new inputs, TensorFlow version

        Args:
            z ((m,d) tf.Variable): *m* input points of dim *d* (same as **x**)
            is_epsilon (boolean): add measurement error
            is_cov (boolean): return covariance matrix

        Returns:
            tuple containing

                - **mean** (*(m,k) tf.Tensor*): *m* mean predictions over **z** for
                    *k* outputs
                - **var** (*(m,m) tf.Tensor*): covariance of **z** inputs
        """
        Kzx = self.tf_Kzx(z, False)  # n_sample(z) x n_sample(x)
        # Kinv_y is n_sample(x) x n_tracks(y), ie indep tracks
        mean = tf.matmul(Kzx, self.Kinv_y)
        var = None
        if is_cov:
            Kzz = self.tf_Kzz(z, is_epsilon)
            var = Kzz - tf.matmul(tf.matmul(Kzx, self.Kinv), tf.transpose(Kzx))
        # mean is n_sample(z) x n_tracks(y), var is n_sample(z) x n_sample(z)
        return (mean, var)

    def tf_predict_random(self, z, is_epsilon=False, cholesky_epsilon=1e-3, n_z=1):
        random_vecs = tf.constant(
            np.float64(np.random.normal(size=(n_z, self.output_dim))),
            dtype='float64')
        mean, var = self.tf_predict(z, is_epsilon, is_cov=True)
        if is_epsilon:
            # stabilise for Cholesky with is_epsilon*I
            var = var + cholesky_epsilon * tf.eye(tf.shape(var)[0], dtype='float64')
        return mean + tf.matmul(tf.linalg.cholesky(var), random_vecs)

    def tf_predict_random_single(self, z, is_epsilon=False):
        random_vecs = tf.constant(np.float64(np.random.normal(size=(1, self.output_dim))), dtype='float64')
        mean, var = self.tf_predict(z, is_epsilon, is_cov=True)
        return mean + tf.sqrt(var) * random_vecs

    def tf_predict_value(self, z, is_epsilon=False, predict_random=False):
        if predict_random:
            return self.tf_predict_random(z, is_epsilon=is_epsilon)
        else:
            return self.tf_predict(z, is_epsilon=is_epsilon, is_cov=False)[0]

    def tf_recursive(self,
                     u_col_tf,
                     x0,
                     is_epsilon=False,
                     predict_random=False,
                     is_nonnegative=False,
                     is_diff=True):
        """Iteratively solve the dynamical system defined by the GP providing
        a (random) transition function as defined by the current GP inputs
        and outputs and ``u_col_tf`` as external forcing. Start with ``x0``
        combined with ``u_col_tf[0,:]`` and along the iteration consider
        input from ``u_col_tf[1:,:]``. Returned are the values of the species
        variables at each simulation step together with the forcing input.

        TODO: in order to mimic the effect of drawing a single function from
        the GP for the whole recursion one would have to constantly add
        previous sampled outputs as new inputs to the GP. This is not done at
        the moment for efficiency's sake, so effectively each iteration draws a
        new transition function from the GP. Future versions should include
        this option.

        Args:
            u_col_tf ((t,d) tf.Tensor): *d*-dim external forcing for *t* steps
            x0 ((k,1) np.array): *k* initial values of species
            is_epsilon: assume measurement error
            predict_random: use GP covariance uncertainty
            is_nonnegative: impose nonnegativity constraint on species
            is_diff: assume the species difference is modeled by GP

        Returns:
            *(t,d+k) tf.Tensor* of forcing input and simulated species,
            first *d* columns return the *u* forcing input
        """
        x_dim = x0.shape[0]
        n_steps, u_dim = u_col_tf.shape
        x_r0 = tf.constant(x0, shape=(1, x_dim), dtype='float64')  # as row vec
        # extend u at time_ind 0 to row vec by x0:
        rtracks = tf.concat([tf.reshape(u_col_tf[0, :], (1, -1)), x_r0], 1)
        #
        # For each step in the simulation
        for t in range(1, n_steps):
            rtracks_prev = tf.reshape(rtracks[t - 1, :],
                                      (1, -1))  # vector to row array
            #
            # Predict next x from previous
            # for prediction need z is (n = 1) x dim array
            x_pred = self.tf_predict_value(rtracks_prev, is_epsilon=is_epsilon, predict_random=predict_random)
            #
            # Are we predicting the change in x or x itself?
            if is_diff:
                x_pred = x_pred + rtracks_prev[:, u_dim:]  # remove U, only X
            if is_nonnegative:
                x_pred = tf.maximum(x_pred, 0)
            #
            # Concatenate x with u and append to rtracks
            rtracks_current = tf.concat(
                [tf.reshape(u_col_tf[t, :], (1, -1)), x_pred], 1)
            rtracks = tf.concat([rtracks, rtracks_current], 0)

        return rtracks


def test_kernel():
    # 2D input
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    # 2-multi output:
    y = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8], [9.9, 10.1]])

    k = Kernel(lengthscales=1.1 * np.ones(x.shape[1]),
               variance=1.2 ** 2,
               likelihood_variance=0.001 ** 2,
               x=x)
    k.set_y(y)

    z = np.array([1, 2, 3, 4, 5, 6], dtype='float64').reshape(-1, x.shape[1])

    # 2D multi output prediction mean, only one covariance:
    m, v = k.np_predict(z)
    assert np.sum(np.abs(m - y[0:3, :])) < 1e-2


def test_kernel_tf():
    sess = tf.Session()

    # --- set up kernel with fixed x and y

    # 2D input
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    # 2-multi output:
    y = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8], [9.9, 10.1]])

    k = Kernel(lengthscales=1.1 * np.ones(x.shape[1]),
               variance=1.2 ** 2,
               likelihood_variance=0.001 ** 2,
               x=x)
    k.set_y(y)

    # ---- set up test variable

    z = tf.Variable([[1, 2], [3, 4]], dtype='float64')
    z
    sess.run(tf.global_variables_initializer())

    m_tf, v_tf = k.tf_predict(z)
    m, v = sess.run([m_tf, v_tf])
    assert np.sum(np.abs(m - y[0:2, :])) < 0.01

    # --- optimise latent test variable with Scipy to fit target distribution
    # ie GPLVM with fixed hyperparameters

    z = tf.Variable([[0.5, 2.5], [3.5, 4.5]], dtype='float64')
    sess.run(tf.global_variables_initializer())

    m_target = np.array([[1.1, 2.2], [3.3, 4.4]], dtype='float64')
    m_tf, v_tf = k.tf_predict(z)
    sess.run(m_tf)  # mean prediction over z far from m_target
    mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(m_tf, v_tf)
    # loss(z) is log prob(m_target | z, k):
    loss = -tf.reduce_sum(mvn.log_prob(m_target))

    # optimise latent z
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss)
    sess.run(loss)
    optimizer.minimize(sess)
    sess.run([loss, z, m_tf])

    z_eval = sess.run(z)
    assert np.sum(np.abs(z_eval - k.x[0:2, :])) < 0.01

    # ----  alternative TF optimizers

    z = tf.Variable([[0.5, 2.5], [3.5, 4.5]], dtype='float64')
    # initialise variables AFTER TF optimizers are set up

    m_target = np.array([[1.1, 2.2], [3.3, 4.4]], dtype='float64')
    m_tf, v_tf = k.tf_predict(z)
    mvn = tf.contrib.distributions.MultivariateNormalFullCovariance(m_tf, v_tf)
    loss = -tf.reduce_sum(mvn.log_prob(m_target))

    # Stepsize absolutely crucial
    # train_op = tf.train.AdamOptimizer(0.05).minimize(loss)
    train_op = tf.train.RMSPropOptimizer(0.05).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    sess.run(tf.global_variables_initializer())  # needed to init optimizer

    sess.run([loss, z, m_tf])
    min_loss = 1e15
    for _ in range(200):
        sess.run(train_op)
        current_loss, current_z, current_m_tf = sess.run([loss, z, m_tf])
        if current_loss < min_loss:
            min_loss = current_loss
            min_result = [current_loss, current_z, current_m_tf]

    min_result
    sess.run([loss, z, m_tf])

    min_z = min_result[1]
    assert np.sum(np.abs(min_z - k.x[0:2, :])) < 0.05
