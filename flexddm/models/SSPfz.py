# packages
from .Model import Model
import numba as nb
import numpy as np
import math
from flexddm import _utilities as util
from functools import partial

"""
Class to simulate data according to the Shrinking Spotlight Model (SSPfz) 
"""


class SSPfz(Model):
    def __init__(
        self,
        data=None,
        input_data_id="PPT",
        input_data_congruency="Condition",
        input_data_rt="RT",
        input_data_accuracy="Correct",
        dt=0.01,
        var=0.1,
        ntrials=100,
        noiseseed=50,
    ):
        """
        Initializes an SSPfz model object.
        """
        self.noiseseed = noiseseed
        self.dt = dt
        self.var = var
        self.ntrials = ntrials

        self.modelsimulationfunction = partial(
            SSPfz.model_simulation,
            dt=self.dt,
            var=self.var,
            nTrials=self.ntrials,
            noiseseed=self.noiseseed,
        )

        if data is not None:
            if isinstance(data, str):
                self.data = util.getRTData(
                    data,
                    input_data_id,
                    input_data_congruency,
                    input_data_rt,
                    input_data_accuracy,
                )
            else:
                self.data = data
            self.bounds = {
                "alpha": (0.07, 0.19),
                "p": (0.2, 0.55),
                "sd_0": (0.1, 2.6),
                "sd_r": (0.01, 0.026),
                "tau": (0.15, min(self.data["rt"])),
            }
        else:
            self.bounds = {
                "alpha": (0.01, 0.19),
                "p": (0.2, 0.55),
                "sd_0": (0.1, 2.6),
                "sd_r": (0.01, 0.026),
                "tau": (0.15, 0.45),
            }

        self.parameter_names = list(self.bounds.keys())
        self.param_number = len(self.parameter_names)

        super().__init__(
            self.param_number, list(self.bounds.values()), self.parameter_names
        )

    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, p, sd_0, sd_r, tau, dt, var, nTrials, noiseseed):
        """
        Performs simulations for SSPfz model.
        @alpha (float): boundary separation
        @beta (float): initial bias
        @p (float): perceptual input of the stimulus
        @sd_0 (float): initial standard deviation of the Gaussian distribution describing the attentional spotlight
        @sd_r (float): shrinking rate of the standard deviation of the Guassian distribution describing the attentional spotlight
        @tau (float): non-decision time
        @dt (float): change in time
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        choicelist = [np.nan] * nTrials
        rtlist = [np.nan] * nTrials

        # Creates congruency list with first half of trials being congruent and the following being incongruent
        congruencylist = ["congruent"] * int(nTrials // 2) + ["incongruent"] * int(
            nTrials // 2
        )

        beta = 0.5

        np.random.seed(noiseseed)
        noise = np.random.normal(loc=0, scale=var, size=10000)
        for n in np.arange(0, nTrials):
            t = tau  # start the accumulation process at non-decision time tau
            evidence = (
                beta * alpha / 2 - (1 - beta) * alpha / 2
            )  # start our evidence at initial-bias beta

            np.random.seed(n)
            while (
                evidence < alpha / 2 and evidence > -alpha / 2
            ):  # keep accumulating evidence until you reach a threshold
                sd = sd_0 - (sd_r * (t - tau))
                if sd <= 0.001:
                    sd = 0.001
                s_ta = ((1 + math.erf((0.5 - 0) / sd / np.sqrt(2))) / 2) - (
                    (1 + math.erf((-0.5 - 0) / sd / np.sqrt(2))) / 2
                )
                s_fl = 1 - s_ta
                if congruencylist[n] == "incongruent":
                    delta = s_ta * p - s_fl * p
                else:
                    delta = s_ta * p + s_fl * p
                evidence += delta * dt + np.random.choice(
                    noise
                )  # add one of the many possible updates to evidence
                t += dt
            if evidence > alpha / 2:
                choicelist[n] = 1  # choose the upper threshold action
            else:
                choicelist[n] = 0  # choose the lower threshold action
            rtlist[n] = t
        return (np.arange(1, nTrials + 1), choicelist, rtlist, congruencylist)
