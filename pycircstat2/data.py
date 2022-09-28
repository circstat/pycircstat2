import json

import numpy as np
import pandas as pd
import pkg_resources


def load_data(
    name,
    source="fisher_1993",
    print_meta=False,
    return_meta=False,
):

    __source__ = ["fisher_1993", "zar_2010", "mardia_1972"]

    # check source
    if source not in __source__:
        raise ValueError(
            f"Invalid source ('{source}').\n Availble sources: {__source__}"
        )

    # load data
    csv_path = pkg_resources.resource_filename(__name__, f"data/{source}/{name}.csv")
    csv_data = pd.read_csv(csv_path, index_col=0)

    # load meta data
    json_path = pkg_resources.resource_filename(
        __name__, f"data/{source}/{name}.csv-metadata.json"
    )
    with open(json_path) as f:
        json_data = json.load(f)

    if print_meta:
        print(json.dumps(json_data, indent=4, ensure_ascii=False))

    if return_meta:
        return csv_data, json_data
    else:
        return csv_data
