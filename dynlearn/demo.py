"""
The module provides functionality to optimise the input to an unknown dynamical
system (but with known or estimated dimension) to achieve a certain production
level for a target species at a specified time point.

"""

import pickle
import numpy as np

from dynlearn import simulation as sf, learn as lf
from dynlearn import get_file_name


def nanog_demo():
    """Optimise input so that a target level of NANOG is produced in the Biomodel
    `Chickarmane2006 - Stem cell switch reversible
    <https://www.ebi.ac.uk/biomodels/BIOMD0000000203>`_
    The model is simulated using a ODE solver of the
    `Tellurium <http://tellurium.analogmachine.org/>`_
    package for biomolecular models.

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
        than can be loaded and displayed by running ``dynlearn/demo_plots.py``

    Call for example under Unix by::

     python3 dynlearn/demo.py
     python3 dynlearn/demo_plots.py
     display dynlearn/results/Nanog_target_50.png

    """
    np.random.seed(123456)  # 123456 with n_samples 10 good

    n_times = 20
    sim = sf.StemCellSwitch(n_times=n_times, real_time=10)

    loss = lf.RegularisedEndLoss(
        target=50.0, target_ind=sim.output_vars.index('NANOG'),
        u_dim=1, time_ind=sim.n_times - 1, reg_weights=0.0)
    gp = lf.FixedGaussGP(
        lengthscales=np.array([100.0 ** 2, 100.0 ** 2, 100.0 ** 2, 20 ** 2,
                               100.0 ** 2, 100.0 ** 2]),
        variance=5 ** 2, likelihood_variance=2 ** 2)

    knots = np.array([0, 5, 10])  # suitable for real_time around 210
    knot_values = np.array([200.0, 150.0, 100.0])
    result_lst = lf.search_u(sim=sim, loss=loss, gp=gp,
                             knots=knots, knot_values=knot_values,
                             x0=np.zeros(len(sim.output_vars)),
                             u_max_limit=1000.0, n_epochs=6-1, n_samples=10)

    file_name = get_file_name('results/result_list_nanog_50_last.dmp')
    with open(file_name, 'wb') as file_ptr:
        pickle.dump(result_lst, file_ptr)
        file_ptr.close()


if __name__ == "__main__":
    nanog_demo()
