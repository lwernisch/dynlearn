"""Simulation of discrete and continuous dynamical systems

:class:`DiscreteSimulation` implements a discrete step solver,
while :class:`ContinuousSimulation` applies the
`Tellurium <http://tellurium.analogmachine.org/>`_ package to solve
ODE models written in the `Antimony
<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2735663/>`_
systems biology model language.

Example:
    Use the subclass :class:`StemCellSwitch` of :class:`ContinuousSimulation`::

        sim = StemCellSwitch(n_times=20, real_time=10.0)
        input_tracks = sim3.u_tracks_from_knots(sim3.n_times, knots=[0,5,10],
                                                knot_values=[500.0,300.0,100.0])
        sim.set_inputs(tracks=input_tracks,
                        time_inds=np.arange(input_tracks.shape[1]))

        sim.dynamic_simulate()
        plt.plot(sim.X.T)

    This class uses a 'step' version of the input, ie, input is held constant
    at the last knot value, eg, at step 7 the value is 300.0.

"""

from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import tellurium as te  # ignore plot error messages
from dynlearn import get_file_name


class Simulation:
    """Generic simulation class"""
    @staticmethod
    def pulse_u_tracks(n_times, knots, knot_values):
        U = np.zeros(n_times)
        U[knots] = knot_values
        return np.reshape(U, (1, -1))

    @staticmethod
    def step_u_tracks(n_times, knots, knot_values):
        U = np.zeros(n_times)
        for k in range(len(knots) - 1):
            U[knots[k]:knots[k + 1]] = knot_values[k]
        U[knots[-1]:] = knot_values[-1]
        return np.reshape(U, (1, -1))

    @abstractmethod
    def u_tracks_from_knots(self, knots, knot_values):
        pass

    @abstractmethod
    def set_inputs(self, tracks, time_inds=None):
        pass

    @abstractmethod
    def dynamic_simulate(self):
        pass


class DiscreteSimulation(Simulation):
    """Solves a dynamical system by iteratively applying a transition
    function ``f_trans`` to the current state values.
    """

    # Args:
    #    n_times: number of simulation steps
    #    u_tracks_from_knots: function to generate complete input from
    #    values at knots (eg as peaks, or as steps)
    #    output_vars: list of species names
    #    x_start: initial values of species
    #    u_type: "peak" or "step"

    def __init__(self, n_times, u_tracks_from_knots,
                 output_vars, x_start, u_type):
        self.n_times = n_times
        self.u_tracks_from_knots = u_tracks_from_knots
        self.output_vars = output_vars
        self.x_start = x_start
        self.u_type = u_type
        self.d = len(x_start)
        self.tr = np.arange(0, self.n_times)
        self.ts = np.linspace(0, self.n_times - 1, self.n_times)

    def inhibit(self, x, alpha, h, k):
        return alpha * k ** h / (k ** h + x ** h)

    def activate(self, x, alpha, h, k):
        return alpha * x ** h / (k ** h + x ** h)

    def set_inputs(self, tracks, time_inds=None):
        self.U = tracks

    def dynamic_simulate(self):
        """Generic discrete solver, applies ``self.f_trans`` iteratively to
        current state and input"""
        X = np.zeros((self.d, self.n_times))
        X[:, 0] = self.x_start
        for t in self.tr[1:]:
            X[:, t] = self.f_trans(X[:, t - 1], self.U[:, t - 1])
        self.X = X


class FeedForwardOrDSimulation(DiscreteSimulation):
    """A discrete version of a feedforward network, details see
    :class:`FeedForwardOrCSimulation`"""

    def __init__(self, n_times):
        DiscreteSimulation.__init__(self, n_times,
                                    u_tracks_from_knots=self.pulse_u_tracks,
                                    output_vars=['U0', 'X1', 'X2'],
                                    x_start=[0, 0, 0], u_type="peak")
        self.f_trans = self.f_trans_ffw
        self.a = 150
        self.ha, self.hi = 5, 5
        self.ka, self.ki = 400, 300
        self.l = 0.25
        self.s = 0.0

    def f_trans_ffw(self, x_old, u):
        x = np.zeros(self.d)
        x[0] = \
            x_old[0] + u[0] - 0.01 * x_old[0] + \
            np.random.normal(0.0, self.s, 1)  # effectively x = u
        x[1] = \
            x_old[1] + self.activate(x_old[0], self.a, self.ha, self.ka) - \
            self.l * x_old[1] + np.random.normal(0.0, self.s, 1)
        x[2] = \
            x_old[2] + self.activate(x_old[0], self.a, self.ha, self.ka) + \
            self.inhibit(x_old[1], self.a, self.hi, self.ki) - \
            self.l * x_old[2] + np.random.normal(0.0, self.s, 1)
        x = np.maximum(x, 0)
        return x


# ---------  Continuous simulations with Antimony and RoadRunner


class ContinuousSimulation(Simulation):
    """
    Uses the `Tellurium <http://tellurium.analogmachine.org/>`_ Antinomy
    model loader and the `roadRunner
    <https://libroadrunner.readthedocs.io/en/latest/>`_ ODE solver to
    simulate dynamical models.
    """
    GENERIC_MODEL_FUNCTIONS = '''
        function inhibit(x,a,h,k)
            a * k^h / (k^h + x^h)
        end
        function activate(x,a,h,k)
           a * x^h / (k^h + x^h)
        end
        '''

    def __init__(self, n_times, u_tracks_from_knots,
                 real_time, output_vars, u_type, model_str=None):
        self.u_tracks_from_knots = u_tracks_from_knots
        self.n_times = n_times
        if real_time is None:
            real_time = n_times
        self.real_time = real_time
        self.output_vars = output_vars
        self.output_dim = len(self.output_vars)
        self.u_type = u_type
        self.preamble_str = self.GENERIC_MODEL_FUNCTIONS
        self.model_str = model_str

    @staticmethod
    @abstractmethod
    def get_input_str(t, d, value):
        # format of the at '(time ...) ...' antimony commmand """
        pass

    def set_inputs(self, tracks, time_inds):
        self.input_times = (time_inds / self.n_times) * self.real_time
        self.input_tracks = tracks
        self.input_str = "\n"
        for i in range(len(self.input_times)):
            u = self.input_tracks.T[i, :, np.newaxis]
            t = self.input_times[i]
            for d in range(u.shape[1]):
                self.input_str += self.get_input_str(t=t, d=d, value=u[d])

    def dynamic_simulate(self):
        if self.model_str is None:
            print("Error simulation: Model definition missing")
            return None
        self.loada_str = self.preamble_str + self.model_str + self.input_str
        self.r = te.loada(self.loada_str)
        self.r.selections = self.output_vars
        self.result = self.r.simulate(0, self.real_time, self.n_times)
        self.X = np.array(self.result).T  # time along shape[1]
        self.U = np.array(self.input_tracks)


class FeedForwardOrCSimulation(ContinuousSimulation):
    """
    A simple example of a regulatory feedforward loop implemented as Antinomy
    model: U0 activates both, X1 and X2, but X2 is inhibited by X1. For
    details see the OR-gate I1 FFL in `Ocone et al,
    2015 <https://www.ncbi.nlm.nih.gov/pubmed/26072513>`_

    Args:
        n_times: number of simulation steps
   """

    def __init__(self, n_times, real_time=None):
        ContinuousSimulation.__init__(self, n_times=n_times,
                                      u_tracks_from_knots=self.pulse_u_tracks,
                                      real_time=real_time,
                                      output_vars=['U0', 'X1', 'X2'],
                                      u_type="peak")
        self.model_str = '''
            g1: -> X1; activate(U0,a,ha,ka)
            g2: -> X2; activate(U0,a,ha,ka) + inhibit(X1,a,hi,ki)
            U0 -> ; lu*U0;
            X1 -> ; l1*X1;
            X2 -> ; l2*X2;
            U0 = 0; X1 = 0; X2 = 0;
            a = 150; ka = 400; ha = 5;
            ki = 300; hi = 5;
            lu = 0.01;
            l1 = 0.25; l2 = 0.25
        '''

    @staticmethod
    def get_input_str(t, d, value):
        # + 1.0 step time delay to make comparable to discrete sim:
        if np.abs(value) > 1e-7:
            # in the GP simulation any input U at time 0 is
            # only felt by X at the next time step, therefore time + 1
            # U is X0 in this simulation ie U/X0 should rise only a time 1
            return "at (time > %f + 1.0): U%i = U%i + %f;\n" % (t, d, d, value)
        else:
            return ""


class StemCellSwitch(ContinuousSimulation):
    """
    Simulation of Biomodel
    `Chickarmane2006 - Stem cell switch reversible
    <https://www.ebi.ac.uk/biomodels/BIOMD0000000203>`_
    The model is simulated using a ODE solver of the
    `Tellurium <http://tellurium.analogmachine.org/>`_
    package for biomolecular models.

    Args:
        n_times: number of simulation steps
        real_time: corresponding simulation time for ODE solver
    """

    def __init__(self, n_times, real_time):
        ContinuousSimulation.__init__(self, n_times=n_times,
                                      u_tracks_from_knots=self.step_u_tracks,
                                      real_time=real_time,
                                      output_vars=['OCT4', 'SOX2', 'NANOG',
                                                   'OCT4_SOX2', 'Protein'],
                                      u_type="step")
        file_name = \
            get_file_name('biomodels/oct4_reversible_0203_template.ant')
        with open(file_name, 'r') as myfile:
            data = myfile.read()
        r = te.loadAntimonyModel(data)
        self.model_str = r.getCurrentAntimony()

    @staticmethod
    def get_input_str(t, d, value):
        # + 1.0 step time delay to make comparable to discrete sim:
        return "at (time > %f): A = %f;\n" % (t, value)


def demo():
    """Demonstrates usage of simulation classes"""

    # -- discrete version of FF

    sim = FeedForwardOrDSimulation(n_times=20)
    u_tracks = sim.u_tracks_from_knots(sim.n_times, knots=[0, 5, 10],
                                       knot_values=[300.0, 200.0, 100.0])
    sim.set_inputs(u_tracks)
    sim.dynamic_simulate()
    plt.clf()
    plt.plot(sim.X.T)

    # -- continuous version of FF

    sim2 = FeedForwardOrCSimulation(n_times=20)
    input_tracks = sim.u_tracks_from_knots(sim.n_times, knots=[0, 5, 10],
                                           knot_values=[300.0, 200.0, 100.0])
    sim2.set_inputs(tracks=input_tracks,
                    time_inds=np.arange(input_tracks.shape[1]))
    sim2.dynamic_simulate()
    plt.clf()
    plt.plot(sim2.X.T)

    # -- compare discrete with continuous version

    np.round(np.hstack([np.array(sim.X.T), np.array(sim2.X.T)]), 2)

    # -- stem cell model

    n_times = 20
    sim3 = StemCellSwitch(n_times=n_times, real_time=10.0)
    input_tracks = \
        sim3.u_tracks_from_knots(sim3.n_times, knots=[0, 5, 10],
                                 knot_values=[581.88, 143.06, 92.76])
    sim3.set_inputs(tracks=input_tracks,
                    time_inds=np.arange(input_tracks.shape[1]))

    sim3.dynamic_simulate()
    plt.clf()
    result = sim3.X.T
    for j in range(result.shape[1]):
        plt.plot(result[:, j], label=sim3.output_vars[j])
    plt.legend()

    import scipy.optimize

    # simple input optimisation example

    n_times = 20
    sim = StemCellSwitch(n_times=n_times, real_time=10.0)
    import numpy as np
    np.arange(20, step=2).dtype
    knots = np.arange(20, step=2)  # suitable for real_time around 20
    x0 = np.array([10.0, 20.0, 30, 40., 50.0, 60.0, 70.0, 80.0, 100, 110])

    def knot_fct(x, knots, sim, name):
        # maximise the final output of species 'name'
        knot_values = np.array(x)
        uvals = \
            sim.u_tracks_from_knots(sim.n_times, knots,
                                    knot_values).T  # uvals always col vector
        input_tracks = uvals.T
        sim.set_inputs(tracks=input_tracks,
                       time_inds=np.arange(input_tracks.shape[1]))
        sim.dynamic_simulate()
        return -sim.X[sim.output_vars.index(name), -1]

    target_name = 'Protein'
    knot_fct(x0, knots, sim, name=target_name)
    res = scipy.optimize.minimize(knot_fct, x0=x0,
                                  args=(knots, sim, 'Protein'),
                                  method="Powell")
    print(res.x)
    print("maximised level of", target_name, "to", -res.fun)
    plt.clf()
    result = sim.X.T
    for j in range(result.shape[1]):
        plt.plot(result[:, j], label=sim3.output_vars[j])
    plt.legend()
