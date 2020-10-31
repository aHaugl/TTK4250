"""

"""
# %% Imports

# types
from typing import (
    Tuple,
    List,
    TypeVar,
    Optional,
    Dict,
    Any,
    Union,
    Sequence,
    Generic,
    Iterable,
)
from mixturedata import MixtureParameters
from gaussparams import GaussParams
from estimatorduck import StateEstimator

# packages
from dataclasses import dataclass
from singledispatchmethod import singledispatchmethod
import numpy as np
from scipy import linalg
from scipy.special import logsumexp

# local
import discretebayes

# %% TypeVar and aliases
MT = TypeVar("MT")  # a type variable to be the mode type

# %% IMM


@dataclass
class IMM(Generic[MT]):
    # The M filters the IMM relies on
    filters: List[StateEstimator[MT]]
    # the transition matrix. PI[i, j] = probability of going from model i to j: shape (M, M)
    PI: np.ndarray

    def __post_init__(self):
        assert (
            self.PI.ndim == 2
        ), "Transition matrix PI shape must be (len(filters), len(filters))"
        assert (
            self.PI.shape[0] == self.PI.shape[1]
        ), "Transition matrix PI shape must be (len(filters), len(filters))"
        assert np.allclose(
            self.PI.sum(axis=1), 1
        ), "The rows of the transition matrix PI must sum to 1."

        assert (
            len(self.filters) == self.PI.shape[0]
        ), "Transition matrix PI shape must be (len(filters), len(filters))"

    def mix_probabilities(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> Tuple[
        np.ndarray, np.ndarray
    ]:  # predicted_mode_probabilities, mix_probabilities: shapes = ((M, (M ,M))).
        # mix_probabilities[s] is the mixture weights for mode s
        """Calculate the predicted mode probability and the mixing probabilities."""

        predicted_mode_probabilities, mix_probabilities = discretebayes.discrete_bayes(
            immstate.weights, self.PI
        )

        assert predicted_mode_probabilities.shape == (
            self.PI.shape[0],
        ), "IMM.mix_probabilities: Wrong shape on the predicted mode probabilities"
        assert (
            mix_probabilities.shape == self.PI.shape
        ), "IMM.mix_probabilities: Wrong shape on mixing probabilities"
        assert np.all(
            np.isfinite(predicted_mode_probabilities)
        ), "IMM.mix_probabilities: predicted mode probabilities not finite"
        assert np.all(
            np.isfinite(mix_probabilities)
        ), "IMM.mix_probabilities: mix probabilities not finite"
        assert np.allclose(
            mix_probabilities.sum(axis=1), 1
        ), "IMM.mix_probabilities: mix probabilities does not sum to 1 per mode"

        return predicted_mode_probabilities, mix_probabilities

    def mix_states(
        self,
        immstate: MixtureParameters[MT],
        # the mixing probabilities: shape=(M, M)
        mix_probabilities: np.ndarray,
    ) -> List[MT]:
        mixed_states = [
            fs.reduce_mixture(MixtureParameters(mix_pr_s, immstate.components))
            for fs, mix_pr_s in zip(self.filters, mix_probabilities)
        ]
        return mixed_states

    def mode_matched_prediction(
        self,
        mode_states: List[MT],
        # The sampling time
        Ts: float,
    ) -> List[MT]:
        modestates_pred = [
            fs.predict(cs, Ts) for fs, cs in zip(self.filters, mode_states)
        ]
        return modestates_pred

    def predict(
        self,
        immstate: MixtureParameters[MT],
        # sampling time
        Ts: float,
    ) -> MixtureParameters[MT]:
        """
        Predict the immstate Ts time units ahead approximating the mixture step

        Ie. Predict mode probabilities, condition states on predicted mode,
        appoximate resulting state distribution as Gaussian for each mode, 
        then predict each mode.
        """

        predicted_mode_prob, mixing_prob = self.mix_probabilities(
            immstate, Ts
        )

        mixed_mode_states: List[MT] = self.mix_states(
            immstate, mixing_prob)

        predicted_mode_states = self.mode_matched_prediction(
            mixed_mode_states, Ts)

        predicted_immstate = MixtureParameters(
            predicted_mode_prob, predicted_mode_states
        )
        return predicted_immstate

    def mode_matched_update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Optional[Dict[str, Any]] = None,
    ) -> List[MT]:
        """Update each mode in immstate with z in sensor_state."""

        updated_states = []
        for filt, gaussparams in zip(self.filters, immstate.components):
            upd = filt.update(z, gaussparams)
            updated_states.append(upd)

        return updated_states

    def update_mode_probabilities(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Calculate the mode probabilities in immstate updated 
        with z in sensor_state"""

        mode_loglikelihood = [
            filt.loglikelihood(z, gaussparam)
            for filt, gaussparam in zip(self.filters, immstate.components)]

        # potential intermediate step logjoint =

        # compute unnormalized first, numerator of eq 6.33
        M = len(immstate.weights)
        updated_mode_probabilities_unnormalized = np.array(
            [mode_loglikelihood[i] + np.log(immstate.weights[i])
             for i in range(M)]
        )

        # normalize so sum(pk) = 1
        # log(p_k) = log(Lambda_k) + log(pk|k-1) + log(a)
        log_a = -logsumexp(updated_mode_probabilities_unnormalized)
        updated_mode_probabilities = np.exp(
            updated_mode_probabilities_unnormalized + log_a)

        # Optional debuging
        assert np.all(np.isfinite(updated_mode_probabilities))
        assert np.allclose(np.sum(updated_mode_probabilities), 1)

        return updated_mode_probabilities

    def update(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Update the immstate with z in sensor_state."""

        updated_weights = self.update_mode_probabilities(
            z, immstate, sensor_state)
        updated_states = self.mode_matched_update(z, immstate, sensor_state)
        updated_immstate = MixtureParameters(updated_weights, updated_states)
        return updated_immstate

    def step(
        self,
        z,
        immstate: MixtureParameters[MT],
        Ts: float,
        sensor_state: Dict[str, Any] = None,
    ) -> MixtureParameters[MT]:
        """Predict immstate with Ts time units followed by updating it 
        with z in sensor_state"""

        predicted_immstate = self.predict(immstate, Ts)
        updated_immstate = self.update(
            z, predicted_immstate, sensor_state=sensor_state)

        return updated_immstate

    def loglikelihood(
        self,
        z: np.ndarray,
        immstate: MixtureParameters,
        *,
        sensor_state: Dict[str, Any] = None,
    ) -> float:

        mode_conditioned_logllike = [
            self.filters[i].loglikelihood(
                z, immstate.components[i], sensor_state=sensor_state)
            for i in range(len(self.filters))
        ]
        logllike = logsumexp(mode_conditioned_logllike, b=immstate.weights)

        return logllike

    def reduce_mixture(
        self, immstate_mixture: MixtureParameters[MixtureParameters[MT]]
    ) -> MixtureParameters[MT]:
        """Approximate a mixture of immstates as a single immstate"""
        # extract probabilities as array
        weights = immstate_mixture.weights

        mode_prob = []
        for sk in range(len(self.filters)):
            mode_prob_sk = 0

            for ak in range(len(immstate_mixture.weights)):
                weights_ak = immstate_mixture.weights[ak]
                mode_prob_sk_giv_ak = (
                    immstate_mixture.components[ak].weights[sk])
                mode_prob_sk += mode_prob_sk_giv_ak * weights_ak

            mode_prob.append(mode_prob_sk)

        mixture_components = []
        for sk in range(len(self.filters)):
            weights = []
            components = []

            for ak in range(len(immstate_mixture.weights)):
                posterior_giv_sk_ak = (
                    immstate_mixture.components[ak].components[sk])
                mode_prob_sk_giv_ak = (
                    immstate_mixture.components[ak].weights[sk])
                weights.append(mode_prob_sk_giv_ak *
                               immstate_mixture.weights[ak] / mode_prob[sk])
                components.append(posterior_giv_sk_ak)

            mixture = MixtureParameters(weights, components)
            # Modes have same reduce function
            reduced = self.filters[0].reduce_mixture(mixture)
            mixture_components.append(reduced)

        reduced = MixtureParameters(mode_prob, mixture_components)

        return reduced

    def estimate(self, immstate: MixtureParameters[MT]) -> GaussParams:
        """Calculate a state estimate with its covariance from immstate"""

        # Modes have same reduce function
        data_reduced = self.filters[0].reduce_mixture(immstate)
        estimate = data_reduced

        return estimate

    def gate(
        self,
        z: np.ndarray,
        immstate: MixtureParameters[MT],
        gate_size_square: float,
        sensor_state: Dict[str, Any] = None,
    ) -> bool:
        """Check if z is within the gate of any mode in immstate 
        in sensor_state"""

        gated_per_mode = [
            self.filters[i].gate(z, immstate.components[i],
                                 gate_size_square, sensor_state=sensor_state)
            for i in range(len(self.filters))
        ]

        gated = True in gated_per_mode
        return gated
