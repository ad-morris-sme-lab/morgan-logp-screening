"""Plotting utilities for model validation."""

from pathlib import Path

from numpy.typing import ArrayLike


def save_parity_plot(observed: ArrayLike, predicted: ArrayLike, path: str | Path) -> Path:
    """Save a labeled observed-versus-predicted parity plot.

    Parameters
    ----------
    observed : array_like
        Held-out target logP values.
    predicted : array_like
        Corresponding predicted logP values.
    path : str or pathlib.Path
        Destination image path. Parent directories are created if needed.

    Returns
    -------
    pathlib.Path
        Resolved output file path.

    Raises
    ------
    ValueError
        If arrays do not have equal nonzero lengths.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from numpy import asarray, isfinite

    y_true, y_pred = asarray(observed, dtype=float), asarray(predicted, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("observed and predicted must be non-empty 1D arrays of equal length")
    if not isfinite(y_true).all() or not isfinite(y_pred).all():
        raise ValueError("plot inputs must be finite")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    low, high = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    if low == high:
        padding = max(abs(float(low)) * 0.05, 0.5)
        low, high = low - padding, high + padding
    fig = Figure(figsize=(5.2, 4.4))
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    axis.scatter(y_true, y_pred, alpha=0.75, edgecolor="white", linewidth=0.35)
    axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    axis.set(xlabel="Computed reference logP", ylabel="Predicted logP", title="Morgan fingerprint model")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    fig.clear()
    return destination.resolve()
