# packages
import numpy as np
import numba as nb
from numba import njit
from numba_stats import norm
from .Model import Model
from flexddm import _utilities as util
from functools import partial


@njit(fastmath=True)
def SSP_drift(t: float, p: float, sda: float, rd: float) -> np.array:
    """Drift function of Spotlight Shrinkage model for Diffusion model in conflict tasks by Evans et al. (2020)

    Args:
        t (float): time of sequential sampling in seconds
        p (float): point of drift rate
        sda (float): inital deviation of drift rate
        rd (float): reduction rate of inital deviation

    Returns:
        np.array: np.array include scalar drift rate
    """
    # calculate current sd of spotlight
    sd_t = sda - (rd * t)
    sd_t = max(sd_t, 0.001)

    x = np.array([0.5], dtype=np.float32)
    # find area of spotlight over target and flanker
    # NOTE numba_stats 1.7.0: norm.cdf only support x is np.array, mu and sigma is float
    a_target = norm.cdf(x, 0.0, sd_t) - norm.cdf(-x, 0.0, sd_t)
    a_flanker = 1 - a_target

    # current drift rate
    drift = p * (a_target - a_flanker)
    # drift = 2 * p * (a_target - 0.5)

    return drift


@njit(fastmath=True)
def DMC_drift(t: float, vc: float, peak: float, shape: float, tau: float) -> float:
    """Drift function for Diffusion model in conflict tasks that follows a gamma function by Evans et al. (2020)

    Arguments
    ---------
        t: float
            Timepoints at which to evaluate the drift.
        vc: float
            The drift for control process.
        shape: float
            Shape parameter of the gamma distribution
        peak: float
            Scalar parameter that scales the peak of
            the gamma distribution.
            (Note this function follows a gamma distribution
            but does not integrate to 1)
        tau: float
            tau is the characteristic time.

    Return
    ------
        float
            The gamma drift evaluated at the supplied timepoints t without congruency condition.
    """

    t = max(t, 0.01)
    term1 = peak * np.exp(-t / tau)
    term2 = np.power((t * np.e) / ((shape - 1) * tau), (shape - 1))
    term3 = ((shape - 1) / t) - (1 / tau)
    va = term1 * term2 * term3

    drift = va + vc
    return drift


class nDDMfz(Model):
    def __init__(
        self,
        data=None,
        input_data_id="PPT",
        input_data_congruency="Condition",
        input_data_rt="RT",
        input_data_accuracy="Correct",
        dt=0.001,
        var=1,
        ntrials=500,
        noiseseed=50,
    ):
        """
        Initializes a standard diffusion model object.
        """

        self.noiseseed = noiseseed
        self.dt = dt
        self.var = var
        self.ntrials = ntrials

        self.modelsimulationfunction = partial(
            nDDMfz.model_simulation,
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
                "$a$": (0.1, 2),
                "$v_{cong}$": (-1, 5),
                "$v_{incong}$": (-1, 5),
                "$t$": (0.1, min(self.data["rt"])),
            }
        else:
            self.bounds = {
                "$a$": (0.1, 2),
                "$v_{cong}$": (-1, 5),
                "$v_{incong}$": (-1, 5),
                "$t$": (0.1, 1),
            }

        self.parameter_names = list(self.bounds.keys())
        self.param_number = len(self.parameter_names)

        super().__init__(
            self.param_number, list(self.bounds.values()), self.parameter_names
        )

    # @staticmethod
    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, delta_c, delta_i, tau, dt, var, nTrials, noiseseed):
        """
        Performs simulations for standard flanker diffusion model.
        @alpha (float): boundary separation
        @delta_c (float): drift rate for incongruent trials
        @delta_i (float): drift rate for incongruent trials
        @tau (float): non-decision time
        @dt (float): change in time
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        beta = 0.5

        choicelist = [np.nan] * nTrials
        rtlist = [np.nan] * nTrials

        congruencylist = ["congruent"] * int(nTrials // 2) + ["incongruent"] * int(
            nTrials // 2
        )
        np.random.seed(noiseseed)
        updates = np.random.normal(loc=0, scale=var, size=10000)

        for n in np.arange(0, nTrials):
            if congruencylist[n] == "congruent":
                delta = delta_c
            else:
                delta = delta_i
            t = dt
            evidence = (
                beta * alpha / 2 - (1 - beta) * alpha / 2
            )  # start our evidence at initial-bias beta
            np.random.seed(n)
            while (
                evidence < alpha / 2 and evidence > -alpha / 2
            ):  # keep accumulating evidence until you reach a threshold
                evidence += delta * dt + np.random.choice(updates) * np.sqrt(dt)
                t += dt  # increment time by the unit dt
            if evidence > alpha / 2:
                    choicelist[n] = 1
            elif evidence < -alpha / 2:
                choicelist[n] = 0
            rtlist[n] = t + tau
        return (np.arange(1, nTrials + 1), choicelist, rtlist, congruencylist)


class nDMCfz(Model):
    def __init__(
        self,
        data=None,
        input_data_id="PPT",
        input_data_congruency="Condition",
        input_data_rt="RT",
        input_data_accuracy="Correct",
        dt=0.001,
        var=1,
        ntrials=500,
        noiseseed=50,
    ):
        """
        Initializes a DMC model object.
        """
        self.noiseseed = noiseseed
        self.dt = dt
        self.var = var
        self.ntrials = ntrials

        self.modelsimulationfunction = noiseseed = partial(
            nDMCfz.model_simulation,
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
                "$a$": (0.7, 1.26),
                "$v_{c}$": (1.58, 6.32),
                "$\alpha$": (1.5, 4.5),
                "$\tau$": (20, 120),
                "$\eta$": (118, 316),
                "$t$": (0.1, min(self.data["rt"])),
            }
        else:
            self.bounds = {
                "$a$": (0.7, 1.26),
                "$v_{c}$": (1.58, 6.32),
                "$\\alpha$": (1.5, 4.5),
                "$\\tau$": (20, 120),
                "$\\eta$": (118, 316),
                "$t$": (0.1, 0.40),
            }
        self.parameter_names = list(self.bounds.keys())
        self.param_number = len(self.parameter_names)

        super().__init__(
            self.param_number, list(self.bounds.values()), self.parameter_names
        )

    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(
        alpha,
        mu_c,
        shape,
        characteristic_time,
        peak_amplitude,
        tau,
        dt,
        var,
        nTrials,
        noiseseed,
    ):
        """
        Performs simulations for DMC model.
        @alpha (float): boundary separation
        @mu_c (float): drift rate of the controlled process
        @shape (float): shape parameter of gamma distribution function used to model the time-course of automatic activation
        @characteristic_time (float): duration of the automatic process
        @peak_amplitude (float): amplitude of automatic activation
        @tau (float): non-decision time
        @dt (float): change in time
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        choicelist = [np.nan] * nTrials
        rtlist = [np.nan] * nTrials

        # Creates congruency list with first half of trials being congruent and the following being incongruent
        congruencylist = ["congruent"] * (nTrials // 2) + ["incongruent"] * (
            nTrials // 2
        )

        beta = 0.5

        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=1000)

        for n in np.arange(0, nTrials):
            t = dt
            evidence = beta * alpha / 2 - (1 - beta) * alpha / 2

            np.random.seed(n)
            while (
                evidence < alpha / 2 and evidence > -alpha / 2
            ):  # keep accumulating evidence until you reach a threshold
                if congruencylist[n] == "congruent":
                    # delta = (
                    #     peak_amplitude
                    #     * np.exp(-(t / characteristic_time))
                    #     * np.power(
                    #         ((t * np.exp(1)) / ((shape - 1) * characteristic_time)),
                    #         (shape - 1),
                    #     )
                    #     * (((shape - 1) / t) - (1 / characteristic_time))
                    # ) + mu_c
                    delta = DMC_drift(
                        t * (1 / dt), mu_c, peak_amplitude, shape, characteristic_time
                    )
                else:
                    # delta = (
                    #     -peak_amplitude
                    #     * np.exp(-(t / characteristic_time))
                    #     * np.power(
                    #         ((t * np.exp(1)) / ((shape - 1) * characteristic_time)),
                    #         (shape - 1),
                    #     )
                    #     * (((shape - 1) / t) - (1 / characteristic_time))
                    # ) + mu_c
                    delta = DMC_drift(
                        t * (1 / dt), mu_c, -peak_amplitude, shape, characteristic_time
                    )
                evidence += delta * dt + np.random.choice(update_jitter) * np.sqrt(dt)
                t += dt  # increment time by the unit dt
                if evidence > alpha / 2:
                    choicelist[n] = 1
                elif evidence < -alpha / 2:
                    choicelist[n] = 0
                rtlist[n] = t + tau

        return (np.arange(1, nTrials + 1), choicelist, rtlist, congruencylist)


class nSSPfz(Model):
    def __init__(
        self,
        data=None,
        input_data_id="PPT",
        input_data_congruency="Condition",
        input_data_rt="RT",
        input_data_accuracy="Correct",
        dt=0.001,
        var=1,
        ntrials=500,
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
            nSSPfz.model_simulation,
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
                "$a$": (1.4, 3.8),
                "$p$": (2, 5.5),
                "$sd_a$": (1, 2.6),
                "$r_d$": (10, 26),
                "$t$": (0.15, min(self.data["rt"])),
            }
        else:
            self.bounds = {
                "$a$": (1.4, 3.8),
                "$p$": (2, 5.5),
                "$sd_a$": (1, 2.6),
                "$r_d$": (10, 26),
                "$t$": (0.15, 0.45),
            }

        self.parameter_names = list(self.bounds.keys())
        self.param_number = len(self.parameter_names)

        super().__init__(
            self.param_number, list(self.bounds.values()), self.parameter_names
        )

    @nb.jit(nopython=True, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(alpha, p, sd_a, sd_r, tau, dt, var, nTrials, noiseseed):
        """
        Performs simulations for SSPfz model.
        @alpha (float): boundary separation
        @p (float): perceptual input of the stimulus
        @sd_a (float): initial standard deviation of the Gaussian distribution describing the attentional spotlight
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
            t = dt
            evidence = (
                beta * alpha / 2 - (1 - beta) * alpha / 2
            )  # start our evidence at initial-bias beta

            np.random.seed(n)
            while (
                evidence < alpha / 2 and evidence > -alpha / 2
            ):  # keep accumulating evidence until you reach a threshold
                # sd = sd_0 - (sd_r * (t - tau))
                # if sd <= 0.001:
                #     sd = 0.001
                # s_ta = ((1 + math.erf((0.5 - 0) / sd / np.sqrt(2))) / 2) - (
                #     (1 + math.erf((-0.5 - 0) / sd / np.sqrt(2))) / 2
                # )
                # s_fl = 1 - s_ta
                # if congruencylist[n] == "incongruent":
                #     delta = s_ta * p - s_fl * p
                # else:
                #     delta = s_ta * p + s_fl * p
                if congruencylist[n] == "incongruent":
                    # NOTE: SSP_drift return array, so we need [0]
                    delta = SSP_drift(t, p, sd_a, sd_r)[0].item()
                else:
                    delta = p
                evidence += delta * dt + np.random.choice(noise) * np.sqrt(dt)
                t += dt
            if evidence > alpha / 2:
                    choicelist[n] = 1
            elif evidence < -alpha / 2:
                choicelist[n] = 0
            rtlist[n] = t + tau

        return (np.arange(1, nTrials + 1), choicelist, rtlist, congruencylist)


class nDSTPfz(Model):
    def __init__(
        self,
        data=None,
        input_data_id="PPT",
        input_data_congruency="Condition",
        input_data_rt="RT",
        input_data_accuracy="Correct",
        dt=0.001,
        var=1,
        ntrials=500,
        noiseseed=50,
    ):
        """
        Initializes a DSTPfz model object.
        """
        self.noiseseed = noiseseed
        self.dt = dt
        self.var = var
        self.ntrials = ntrials

        self.modelsimulationfunction = partial(
            nDSTPfz.model_simulation,
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
                "$a_{ss}$": (1.4, 3.8),
                "$v_{ss}$": (2.5, 5.5),
                "$a$": (1.4, 3.8),
                "$v_{ta}$": (0.5, 1.5),
                "$v_{fl}$": (0.5, 2.5),
                "$v_{p2}$": (4.0, 12.0),
                "$t$": (0.15, min(self.data["rt"])),
            }
        else:
            self.bounds = {
                "$a_{ss}$": (1.4, 3.8),
                "$v_{ss}$": (2.5, 5.5),
                "$a$": (1.4, 3.8),
                "$v_{ta}$": (0.5, 1.5),
                "$v_{fl}$": (0.5, 2.5),
                "$v_{p2}$": (4.0, 12.0),
                "$t$": (0.15, 0.45),
            }

        self.parameter_names = list(self.bounds.keys())
        self.param_number = len(self.parameter_names)

        super().__init__(
            self.param_number, list(self.bounds.values()), self.parameter_names
        )

    @nb.jit(nopython=False, cache=True, parallel=False, fastmath=True, nogil=True)
    def model_simulation(
        alphaSS,
        deltaSS,
        alphaRS,
        delta_target,
        delta_flanker,
        deltaRS,
        tau,
        dt,
        var,
        nTrials,
        noiseseed,
    ):
        """
        Performs simulations for DSTPfz model.
        @alphaSS (float): boundary separation for stimulus selection phase
        @betaSS (float): initial bias for stimulus selection phase
        @deltaSS (float): drift rate for stimulus selection phase
        @alphaRS (float): boundary separation for response selection phase
        @betaRS (float): inital bias for response selection phase
        @delta_target (float): drift rate for target arrow during response selection BEFORE stimulus is selected
        @delta_flanker (float): drift rate for flanker arrows during response selection BEFORE stimulus is selected
        @deltaRS (float): drift rate for the reponse selection phase after a stimulus (either flanker or target) has been selected
        @tau (float): non-decision time
        @dt (float): change in time
        @var (float): variance
        @nTrials (int): number of trials
        @noiseseed (int): random seed for noise consistency
        """

        betaSS = 0.5
        betaRS = 0.5

        choicelist = [np.nan] * nTrials
        rtlist = [np.nan] * nTrials

        congruencylist = ["congruent"] * int(nTrials // 2) + ["incongruent"] * int(
            nTrials // 2
        )

        np.random.seed(noiseseed)
        update_jitter = np.random.normal(loc=0, scale=var, size=(10000, 2))
        for n in np.arange(0, nTrials):
            if congruencylist[n] == "congruent":
                drift = delta_target + delta_flanker
            else:
                drift = delta_target - delta_flanker
            t = dt

            # start our evidence at initial-bias beta (Kyle: I modified this so beta is always between 0 and 1, and alpha is the total distance between bounds)
            evidenceSS = betaSS * alphaSS / 2 - (1 - betaSS) * alphaSS / 2
            evidenceRS = betaRS * alphaRS / 2 - (1 - betaRS) * alphaRS / 2

            np.random.seed(n)
            while evidenceRS < alphaRS / 2 and evidenceRS > -alphaRS / 2:
                # Stimulus selection
                evidenceSS += deltaSS * dt + np.random.choice(
                    update_jitter[:, 0]
                ) * np.sqrt(dt)
                # Drift rate in the second phase
                if evidenceSS > alphaSS / 2:  # select target
                    drift = deltaRS
                elif evidenceSS < -alphaSS / 2:  # select flanker
                    # the flanker have same direction as to target or error
                    drift = deltaRS if congruencylist[n] == "congruent" else -deltaRS

                # DDM equation
                evidenceRS += drift * dt + np.random.choice(
                    update_jitter[:, 1]
                ) * np.sqrt(dt)
                t += dt

            if evidenceRS > alphaRS / 2:
                choicelist[n] = 1  # choose the upper threshold action
            elif evidenceRS < -alphaRS / 2:
                choicelist[n] = 0  # choose the lower threshold action
            rtlist[n] = t + tau

        return (np.arange(1, nTrials + 1), choicelist, rtlist, congruencylist)
