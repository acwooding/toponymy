import os
from warnings import warn
import numpy as np
import numba


def resolve_api_key(
    api_key: str | None,
    env_new: str | None,
    env_legacy: str | None = None,
    required: bool = True,
) -> str | None:
    """
    Resolve API key from explicit parameter or environment variables.

    Parameters
    ----------
    api_key : str | None
        Explicitly provided API key (takes precedence over environment variables).
    env_new : str | None
        Primary environment variable name to check (e.g., "COHERE_API_KEY").
    env_legacy : str | None, optional
        Deprecated environment variable name for backwards compatibility.
        If found, a deprecation warning is issued.
    required : bool, default=True
        If True, raises ValueError when no API key is found.
        If False, returns None when no API key is found.

    Returns
    -------
    str | None
        The resolved API key, or None if not found and not required.

    Raises
    ------
    ValueError
        If required=True and no API key is found.
    """
    if api_key is not None:
        return api_key

    new_key = os.getenv(env_new) if env_new else None
    legacy_key = os.getenv(env_legacy) if env_legacy else None

    if new_key:
        return new_key

    if legacy_key:
        warn(
            f"{env_legacy} is deprecated. Use {env_new} instead.",
            FutureWarning,
            stacklevel=3,
        )
        return legacy_key

    if required:
        # Extract readable provider name from env variable (e.g., "COHERE_API_KEY" -> "Cohere")
        provider_name = (
            env_new.replace("_API_KEY", "").replace("_", " ").title()
            if env_new
            else "API"
        )
        raise ValueError(
            f"No {provider_name} API key provided. Set {env_new} environment variable "
            f"or pass api_key parameter."
        )

    return None


@numba.njit(fastmath=True, cache=True)
def distance_to_vector(vector, other_vectors):
    result = np.zeros(other_vectors.shape[0], dtype=np.float32)
    other_vector_norms = np.zeros(other_vectors.shape[0], dtype=np.float32)
    vector_norm = 0.0
    for j in range(vector.shape[0]):
        vector_norm += vector[j] * vector[j]

    for i in range(other_vectors.shape[0]):
        for j in range(vector.shape[0]):
            result[i] += vector[j] * other_vectors[i, j]
            other_vector_norms[i] += other_vectors[i, j] * other_vectors[i, j]

    if vector_norm == 0.0:
        return np.ones(other_vectors.shape[0], dtype=np.float64)
    else:
        return 1.0 - (result / np.sqrt(vector_norm * other_vector_norms))


@numba.njit(cache=True)
def diversify_fixed_alpha(query_vector, candidate_neighbor_vectors, alpha=1.0):
    distance_to_query = distance_to_vector(query_vector, candidate_neighbor_vectors)

    retained_neighbor_indices = [0]
    for i, vector in enumerate(candidate_neighbor_vectors[1:], 1):
        retained_vectors = candidate_neighbor_vectors[
            np.array(retained_neighbor_indices)
        ]
        retained_neighbor_distances = distance_to_vector(
            vector,
            retained_vectors,
        )
        for j in range(retained_neighbor_distances.shape[0]):
            if alpha * distance_to_query[i] > retained_neighbor_distances[j]:
                break
        else:
            retained_neighbor_indices.append(i)

    return retained_neighbor_indices


@numba.njit(cache=True)
def diversify_max_alpha(
    query_vector,
    candidate_neighbor_vectors,
    n_results,
    max_alpha=1.0,
    min_alpha=0.0,
    tolerance=0.01,
):
    mid_alpha = (max_alpha + min_alpha) / 2.0

    while abs(max_alpha - min_alpha) > tolerance:
        results = diversify_fixed_alpha(
            query_vector, candidate_neighbor_vectors, alpha=mid_alpha
        )
        if len(results) >= n_results:
            min_alpha = mid_alpha
        else:
            max_alpha = mid_alpha

        mid_alpha = (min_alpha + max_alpha) / 2.0

    return diversify_fixed_alpha(
        query_vector, candidate_neighbor_vectors, alpha=min_alpha
    )
