"""Sun position (NOAA solar calculator algorithm, accurate to about 0.01 degree)."""

import datetime as dt

import numpy as np


def solar_position(times_utc: np.ndarray, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Azimuth (degrees clockwise from true north) and apparent elevation (degrees, with atmospheric
    refraction) of the sun.

    Args:
        times_utc: ``datetime64`` instants in UTC.
        lat, lon: Observer latitude and longitude in degrees.
    """
    seconds = np.asarray(times_utc, dtype="datetime64[s]").astype("int64").astype("float64")
    jc = (seconds / 86400.0 + 2440587.5 - 2451545.0) / 36525.0
    rad = np.radians

    mean_long = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    centre = (
        np.sin(rad(mean_anom)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(rad(2 * mean_anom)) * (0.019993 - 0.000101 * jc)
        + np.sin(rad(3 * mean_anom)) * 0.000289
    )
    omega = rad(125.04 - 1934.136 * jc)
    app_long = mean_long + centre - 0.00569 - 0.00478 * np.sin(omega)
    mean_obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq = rad(mean_obliq + 0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(obliq) * np.sin(rad(app_long)))

    var_y = np.tan(obliq / 2) ** 2
    l0, m = rad(mean_long), rad(mean_anom)
    eq_time = 4 * np.degrees(
        var_y * np.sin(2 * l0)
        - 2 * ecc * np.sin(m)
        + 4 * ecc * var_y * np.sin(m) * np.cos(2 * l0)
        - 0.5 * var_y**2 * np.sin(4 * l0)
        - 1.25 * ecc**2 * np.sin(2 * m)
    )
    minutes = np.mod(seconds, 86400.0) / 60.0
    true_solar = np.mod(minutes + eq_time + 4 * lon, 1440.0)
    hour_angle = rad(true_solar / 4 - 180)

    phi = rad(lat)
    cos_zen = np.clip(np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle), -1, 1)
    zenith = np.arccos(cos_zen)
    elevation = 90 - np.degrees(zenith)

    with np.errstate(invalid="ignore", divide="ignore"):
        cos_az = (np.sin(phi) * cos_zen - np.sin(decl)) / (np.cos(phi) * np.sin(zenith))
    az = np.degrees(np.arccos(np.clip(np.nan_to_num(cos_az), -1, 1)))
    azimuth = np.where(hour_angle > 0, np.mod(az + 180, 360), np.mod(540 - az, 360))

    return azimuth, elevation + _refraction(elevation)


def _refraction(elevation: np.ndarray) -> np.ndarray:
    """Atmospheric refraction in degrees (NOAA approximation)."""
    e = np.asarray(elevation, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.tan(np.radians(e))
        arcsec = np.select(
            [e > 85, e > 5, e > -0.575],
            [
                0.0,
                58.1 / t - 0.07 / t**3 + 0.000086 / t**5,
                1735 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711))),
            ],
            -20.772 / t,
        )
    return arcsec / 3600.0


def day_samples(date: dt.date, lon: float, step_min: float) -> np.ndarray:
    """Midpoints of ``step_min``-minute intervals covering one solar day (local mean solar midnight to
    midnight at longitude ``lon``), as UTC ``datetime64``."""
    n = int(round(1440 / step_min))
    if n < 1 or abs(n * step_min - 1440) > 1e-6:
        raise ValueError("time_step_min must divide a day (1440 minutes) evenly")
    midnight = np.datetime64(date.isoformat(), "s") - np.timedelta64(int(round(lon / 15 * 3600)), "s")
    offsets = ((np.arange(n) + 0.5) * step_min * 60).round().astype("int64")
    return midnight + offsets.astype("timedelta64[s]")
