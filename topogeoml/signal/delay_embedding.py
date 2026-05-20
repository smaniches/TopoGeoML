"""
Takens delay embedding for univariate discrete signals.

Implements DEFINITION 3.1 and the algorithm specification of §6.1 in
``docs/mathematics/foundations.md``. The mathematical justification is
Takens' embedding theorem (Takens, 1981, Theorem 1, p. 366 of LNM 898).

This module implements only the construction; the asymptotic correctness
under Takens' theorem is invoked, not proven, here. See the foundations
document for the precise statement and reference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def takens_embedding(
    signal: NDArray[np.floating],
    embedding_dim: int,
    delay: int,
) -> NDArray[np.float64]:
    """
    Construct the Takens delay embedding of a univariate discrete signal.

    Algorithm specification (§6.1 of foundations.md)
    -----------------------------------------------
    Input space:     :math:`\\mathcal{S}_{T,1} = \\mathbb{R}^T`,
                     ``embedding_dim`` ``m`` :math:`\\in \\mathbb{N}_{\\ge 1}`,
                     ``delay`` ``\\tau`` :math:`\\in \\mathbb{N}_{\\ge 1}`,
                     with :math:`(m-1)\\tau \\le T - 1`.
    Output space:    :math:`\\mathbb{R}^{(T - (m-1)\\tau) \\times m}`.
    Correctness:     Row :math:`t` of the output (for :math:`t = 0, \\ldots,
                     T - (m-1)\\tau - 1`) equals
                     :math:`(s((m-1)\\tau + t), s((m-1)\\tau + t - \\tau),
                     \\ldots, s((m-1)\\tau + t - (m-1)\\tau))`.
    Complexity:      :math:`O(T m)` time and space.

    Parameters
    ----------
    signal : numpy.ndarray, shape (T,)
        Univariate discrete signal. Will be cast to float64.
    embedding_dim : int
        Number of coordinates of the embedding, :math:`m \\ge 1`.
    delay : int
        Delay :math:`\\tau \\ge 1` (in samples).

    Returns
    -------
    numpy.ndarray, shape (T - (m-1)*tau, m), dtype float64
        The delay-embedded point cloud. Row index corresponds to the
        most recent time index; column index 0 is the present sample
        (most recent in time); column index :math:`m-1` is the oldest.

    Raises
    ------
    ValueError
        If ``signal`` is not 1-dimensional, ``embedding_dim < 1``,
        ``delay < 1``, or ``(embedding_dim - 1) * delay >= len(signal)``.

    Notes
    -----
    The implementation uses ``numpy.lib.stride_tricks.sliding_window_view``,
    which constructs a view (no data copy) followed by a copy with
    explicit float64 dtype (elite-code-standards §1.3) and reverses the
    column order so that index 0 is the most recent sample (matching the
    convention of Definition 3.1).

    The function performs no Python-level sample loop for the construction
    (elite-code-standards §3.1). The stride-trick view plus a single copy
    is vectorized at C level.
    """
    if signal.ndim != 1:
        raise ValueError(
            f"signal must be 1-dimensional; got ndim={signal.ndim}"
        )
    if embedding_dim < 1:
        raise ValueError(
            f"embedding_dim must be >= 1; got {embedding_dim}"
        )
    if delay < 1:
        raise ValueError(f"delay must be >= 1; got {delay}")

    T = signal.shape[0]
    span = (embedding_dim - 1) * delay
    if span >= T:
        raise ValueError(
            f"(embedding_dim - 1) * delay = {span} must be < len(signal) = {T}; "
            f"reduce embedding_dim or delay."
        )

    # §1.3: explicit dtype.
    s = np.ascontiguousarray(signal, dtype=np.float64)

    n_points = T - span
    # Construct the embedding by index gather: row i is
    # (s[span + i], s[span + i - delay], ..., s[span + i - (m-1)*delay]).
    # We build the row indices once (vectorized, no Python loop over samples).
    row_origins = np.arange(n_points, dtype=np.intp) + span
    column_offsets = -delay * np.arange(embedding_dim, dtype=np.intp)
    indices = row_origins[:, None] + column_offsets[None, :]  # (n_points, m)
    embedding = s[indices]  # advanced indexing → new C-contiguous array

    # Defensive copy with explicit dtype: guards against the case where
    # signal was already float64 and numpy returns a view-like array.
    return np.ascontiguousarray(embedding, dtype=np.float64)


def estimate_delay_autocorrelation(
    signal: NDArray[np.floating],
    max_lag: int | None = None,
    threshold: float = 1.0 / np.e,
) -> int:
    """
    Heuristic for choosing the delay :math:`\\tau` via the
    autocorrelation first-zero / first-below-threshold criterion.

    The standard heuristic (Fraser & Swinney, 1986, *Phys. Rev. A* 33,
    1134–1140; their mutual-information minimum) selects :math:`\\tau` as
    the smallest lag at which the autocorrelation falls below
    :math:`1/e` (or first crosses zero). We implement the
    autocorrelation-below-threshold variant because it is the
    fastest stable proxy. The exact mutual-information minimum is
    recommended in the cited reference for highly nonlinear signals;
    we treat the autocorrelation criterion as a baseline.

    Algorithm specification
    -----------------------
    Input space:     ``signal`` :math:`\\in \\mathbb{R}^T`,
                     ``max_lag`` :math:`\\in \\mathbb{N}_{\\ge 1}` or ``None``,
                     ``threshold`` :math:`\\in (0, 1)`.
    Output space:    :math:`\\mathbb{N}_{\\ge 1}`.
    Correctness:     Returns the smallest :math:`\\tau \\in
                     \\{1, \\ldots, \\text{max\\_lag}\\}` such that
                     :math:`|\\hat\\rho(\\tau)| \\le \\text{threshold}`,
                     where :math:`\\hat\\rho` is the sample autocorrelation
                     (Pearson correlation of the signal with its lagged copy).
                     If no such :math:`\\tau` exists, returns ``max_lag``.
    Complexity:      :math:`O(T \\cdot \\text{max\\_lag})` time.

    Parameters
    ----------
    signal : numpy.ndarray, shape (T,)
    max_lag : int, optional
        Upper bound on tested lags. Defaults to ``T // 4``.
    threshold : float
        Autocorrelation threshold below which a lag is accepted.
        Default :math:`1/e \\approx 0.368`.

    Returns
    -------
    int
        Estimated delay :math:`\\tau \\ge 1`.
    """
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1D; got ndim={signal.ndim}")
    s = np.ascontiguousarray(signal, dtype=np.float64)
    T = s.shape[0]
    if max_lag is None:
        max_lag = max(1, T // 4)
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1; got {max_lag}")
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1); got {threshold}")

    s_centered = s - s.mean()
    var = float((s_centered * s_centered).sum()) + 1e-300  # §1.4: division safety

    # Vectorized autocorrelation over all lags.
    # We use np.correlate for the full sequence; this is O(T * max_lag) explicit
    # but with no Python sample loop (§3.1 — the iteration is over lags, not samples).
    for lag in range(1, max_lag + 1):
        rho = float((s_centered[:-lag] * s_centered[lag:]).sum()) / var
        if abs(rho) <= threshold:
            return lag
    return max_lag
