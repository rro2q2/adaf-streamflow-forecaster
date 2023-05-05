import sys
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class CamelsCL(BaseDataset):
    """Data set class for the CAMELS CL dataset by [#]_.
    For more efficient data loading during model training/evaluating, this dataset class expects the CAMELS-CL dataset
    in a processed format. Specifically, this dataset class works with per-basin csv files that contain all 
    timeseries data combined. Use the :func:`preprocess_camels_cl_dataset` function to process the original dataset 
    layout into this format.
    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    References
    ----------
    .. [#] Alvarez-Garreton, C., Mendoza, P. A., Boisier, J. P., Addor, N., Galleguillos, M., Zambrano-Bigiarini, M.,
        Lara, A., Puelma, C., Cortes, G., Garreaud, R., McPhee, J., and Ayala, A.: The CAMELS-CL dataset: catchment
        attributes and meteorology for large sample studies - Chile dataset, Hydrol. Earth Syst. Sci., 22, 5817-5846,
        https://doi.org/10.5194/hess-22-5817-2018, 2018.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CamelsCL, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        # get forcings
        dfs = []
        for forcing in self.cfg.forcings:
            df, area = load_camels_cl_timeseries(self.cfg.data_dir, basin, forcing)

            # rename columns
            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)

        # add discharge
        df['QObs(mm/d)'] = load_camels_cl_discharge(self.cfg.data_dir, basin)

        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if "qobs" in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load static catchment attributes."""
        return load_camels_cl_attributes(self.cfg.data_dir, basins=self.basins)


def load_camels_cl_timeseries(data_dir: Path, basin: str, forcings: str) -> pd.DataFrame:
    """Load the time series data for one basin of the CAMELS CL data set.
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS CL directory. This folder must contain a folder called 'preprocessed' containing the 
        per-basin csv files created by :func:`preprocess_camels_cl_dataset`.
    basin : str
        Basin identifier number as string.
    forcings : str
        Focrcings path as a string to load the timeseries data
    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
    int
        Catchment area (m2), specified in the header of the forcing file.
    Raises
    ------
    FileNotFoundError
        If no basin exists within the file path of the CAMELS CL dataset directory.
    """
    forcing_path = data_dir / 'basin_mean_forcing' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    file_path = list(forcing_path.glob(f'**/{basin}_*_forcing_leap.txt'))

    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    with open(file_path, 'r') as fp:
        # load area from header
        fp.readline()
        fp.readline()
        area = int(fp.readline())
        # load the dataframe from the rest of the stream
        df = pd.read_csv(fp, sep='\s+')
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
                                    format="%Y/%m/%d")
        df = df.set_index("date")

    return df, area



def load_camels_cl_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS CL attributes
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS CL directory. Assumes that a folder called 'chiles_attributes' exists.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.
    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    attributes_path = data_dir / 'chile_attributes'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_cl_*.txt')


    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    
    # convert huc column to double digit strings
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df



def load_camels_cl_discharge(data_dir: Path, basin: str):
    """Load the discharge data for a basin of the CAMELS CL data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-CL data set. All txt-files from the original dataset should be present in this folder. 
    basin : str
        Basin identifier numbered as a string.
    Raises
    ------
    FileExistsError
        If no basin exists within the file path of the CAMELS CL directory.
    """
    discharge_path = data_dir / 'chile_streamflow'
    file_path = list(discharge_path.glob(f'**/{basin}_streamflow_qc.txt'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
    df = df.set_index("date")

    return df.QObs

