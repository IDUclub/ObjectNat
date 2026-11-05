import numpy as np

from objectnat import config

from .noise_init_data import air_resist_ratio

logger = config.logger

MAX_DB_VALUE = 194  # the loudest undistorted sound in air at sea level, dB

_TABLE = air_resist_ratio.sort_index().sort_index(axis=1)
_FREQS = _TABLE.index.to_numpy(dtype=float)
_TEMPS = _TABLE.columns.to_numpy(dtype=float)
_VALUES = _TABLE.to_numpy(dtype=float)


def get_air_resist_ratio(temp, freq, check_temp_freq=False):
    """
    Return the air absorption coefficient for temperature and frequency.

    Values are looked up in the bundled ``air_resist_ratio`` table. Exact
    temperature/frequency pairs are returned directly; in-between values are
    linearly interpolated across one or both axes. Values outside the table are
    clamped to the nearest available boundary. When ``check_temp_freq`` is
    true, out-of-range values also produce logger warnings.

    Args:
        temp:
            Air temperature in degrees Celsius.
        freq:
            Geometric mean frequency in hertz.
        check_temp_freq:
            Whether to warn when ``temp`` or ``freq`` is outside the tabulated
            range.

    Returns:
        float:
            Air absorption coefficient used by the distance attenuation model.
    """

    if check_temp_freq:
        if temp > max(air_resist_ratio.columns) or temp < min(air_resist_ratio.columns):
            logger.warning(
                f"The specified temperature of {temp}°C is outside the tabulated data range. "
                f"The air resistance coefficient for these values may be inaccurate. "
                f"Recommended temperature range: {min(air_resist_ratio.columns)}°C "
                f"to {max(air_resist_ratio.columns)}°C."
            )

        if freq > max(air_resist_ratio.index) or freq < min(air_resist_ratio.index):
            logger.warning(
                f"The specified geometric mean frequency of {freq} Hz is outside the tabulated data range."
                f" The air resistance coefficient for these values may be inaccurate."
                f" Recommended frequency range: {min(air_resist_ratio.index)} Hz to {max(air_resist_ratio.index)} Hz."
            )

    # bilinear interpolation, clamped to the table edges
    by_temperature = [np.interp(freq, _FREQS, _VALUES[:, k]) for k in range(len(_TEMPS))]
    return float(np.interp(temp, _TEMPS, by_temperature))


def decay_distance(level_db: float, target_db: float, air_db_per_m: float, *, line: bool = False) -> float:
    """
    Distance in metres at which a source of ``level_db`` (at 1 m) falls to ``target_db``: the root of the point law
    ``L - 20 lg r - k r`` or, with ``line=True``, of the line law ``L - 10 lg r - k r`` (``k`` in dB per metre).
    """
    spread = 10.0 if line else 20.0
    lo, hi = -9.0, 7.0  # log10 of the distance: the level falls monotonically, so the root is in between
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if level_db - spread * mid - air_db_per_m * 10**mid > target_db:
            lo = mid
        else:
            hi = mid
    return float(10**lo)


def dist_to_target_db(
    init_noise_db, target_noise_db, geometric_mean_freq_hz, air_temperature, return_desc=False, check_temp_freq=False
) -> float | str:
    """
    Calculates the distance required for a sound wave to decay from an initial noise level to a target noise level,
    based on the geometric mean frequency of the sound and the air temperature. Optionally, can return a description
    of the sound propagation behavior.

    Args:
        init_noise_db (float): The initial noise level of the source in decibels (dB). This is the starting sound
            intensity.
        target_noise_db (float): The target noise level in decibels (dB), representing the level to which the sound
            decays over distance.
        geometric_mean_freq_hz (float): The geometric mean frequency of the sound (in Hz). This frequency influences
            the attenuation of sound over distance. Higher frequencies decay faster than lower ones.
        air_temperature (float): The temperature of the air in degrees Celsius. This influences the air's resistance
            to sound propagation.
        return_desc (bool, optional): If set to `True`, the function will return a description of the sound decay
            process instead of the calculated distance.
        check_temp_freq (bool, optional): If `True`, the function will check whether the temperature and frequency
            are within valid ranges.

    Returns:
        float or str: If `return_desc` is `False`, the function returns the distance (in meters) over which the sound
        decays from `init_noise_db` to `target_noise_db`. If `return_desc` is `True`, a descriptive string is returned
        explaining the calculation and the conditions.
    """

    k = get_air_resist_ratio(air_temperature, geometric_mean_freq_hz, check_temp_freq)
    distance = decay_distance(init_noise_db, target_noise_db, k)
    if return_desc:
        return (
            f"Noise level of {init_noise_db} dB "
            f"with a geometric mean frequency of {geometric_mean_freq_hz} Hz "
            f"at an air temperature of {air_temperature}°C decays to {target_noise_db} dB "
            f"over a distance of {distance} meters. Air resistance coefficient: {k}."
        )
    return distance


def source_values(sources, column: str, default: float | None) -> np.ndarray:
    """
    Per-source ``source_noise_db`` or ``geometric_mean_freq_hz``: the column of ``sources`` when it exists,
    otherwise ``default`` for every source. Levels above the physical limit in air are rejected.
    """
    if column in sources.columns:
        if sources[column].isnull().any():
            raise ValueError(f"Column '{column}' contains missing (NaN) values")
        values = sources[column].to_numpy(dtype=float)
    elif default is None:
        raise ValueError(f"Either `{column}` must be provided or the sources must contain a '{column}' column.")
    else:
        values = np.full(len(sources), float(default))
    if column == "source_noise_db" and (values > MAX_DB_VALUE).any():
        raise ValueError(f"source_noise_db exceeds the physical limit of {MAX_DB_VALUE} dB in air.")
    return values


def vegetation_db_per_m(geometric_mean_freq_hz: float) -> float:
    """Attenuation by foliage in dB per metre of vegetation crossed, growing with the cube root of the frequency."""
    return 0.08 * geometric_mean_freq_hz ** (1 / 3) / 8


def green_noise_reduce_db(geometric_mean_freq_hz, r_tree) -> float:
    """
    Calculates the amount of noise reduction (in dB) provided by vegetation of a given thickness at a specified
    geometric mean frequency. The function models the reduction based on the interaction of the sound with trees or
    vegetation.

    Args:
        geometric_mean_freq_hz (float): The geometric mean frequency of the sound (in Hz).
        r_tree (float): The thickness or density of the vegetation (in meters).

    Returns:
        float: The noise reduction (in dB) achieved by the vegetation. This value indicates how much quieter the sound
        will be after passing through or interacting with the vegetation of the specified thickness.
    """
    return round(vegetation_db_per_m(geometric_mean_freq_hz) * r_tree, 1)
