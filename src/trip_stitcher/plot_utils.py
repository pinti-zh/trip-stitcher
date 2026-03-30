from collections import OrderedDict
from colorsys import hsv_to_rgb

import numpy as np
from sklearn.decomposition import PCA

from trip_stitcher.models import Stop


def color_map(h_value: float) -> tuple[float, float, float]:
    h_value = max(0.0, min(1.0, h_value))
    return hsv_to_rgb(h_value, 1.0, 1.0)


def name_to_line_map(stops: list[Stop]) -> dict[str, float]:
    data_dict = OrderedDict()
    for stop in stops:
        if stop.name not in data_dict:
            data_dict[stop.name] = {
                "lat": stop.lat,
                "lon": stop.lon,
            }

    lats = np.array([value_dict["lat"] for value_dict in data_dict.values()])
    lats = (lats - lats.min()) / (lats.max() - lats.min())
    lons = np.array([value_dict["lon"] for value_dict in data_dict.values()])
    lons = (lons - lons.min()) / (lons.max() - lons.min())

    data = np.array([[lat, lon] for lat, lon in zip(lats, lons)])
    pca = PCA(n_components=1)
    data_pca = pca.fit_transform(data).flatten()
    data_pca -= data_pca.min()
    data_pca /= data_pca.max()

    idx = np.argsort(data_pca)
    stop_names = np.array(list(data_dict.keys()))[idx]
    sorted_values = data_pca[idx]

    diffs = np.where(
        np.diff(sorted_values) < (0.5 / len(sorted_values)), 0.5 / len(sorted_values), np.diff(sorted_values)
    )
    log_diffs = np.log(1 + 10 * diffs)
    values = [sorted_values[0]]
    for log_diff in log_diffs:
        values.append(values[-1] + log_diff)
    values = np.array(values) / max(values)

    return dict((stop_name, value) for stop_name, value in zip(stop_names, values))
