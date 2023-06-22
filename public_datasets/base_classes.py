import os
import re
from glob import glob
import pandas as pd
import polars as pl
from loguru import logger

from utils.string import rreplace

MODAL_PATH_PATTERN = '{root}/{modal}'
PARQUET_PATH_PATTERN = MODAL_PATH_PATTERN + '/subject_{subject}/{session}.parquet'


class ParquetDatasetFormatter:
    """
    This class processes raw dataset and save as parquet files in a structured directory.
    """

    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict):
        """
        This class transforms public datasets into the same format for ease of use.

        Args:
            raw_folder: path to unprocessed dataset
            destination_folder: folder to save output
            sampling_rates: a dict containing sampling rates of each modal to resample by linear interpolation.
                - key: modal name
                - value: sampling rate (unit: Hz)
        """
        self.raw_folder = raw_folder
        self.destination_folder = destination_folder

        # convert Hz to sample/msec
        self.sampling_rates = {k: v / 1000 for k, v in sampling_rates.items()}

    def get_output_file_path(self, modal, subject, session) -> str:
        """
        Get path to an output file (.parquet)

        Args:
            modal: modality
            subject: subject ID
            session: session ID

        Returns:
            path to parquet file
        """
        p = PARQUET_PATH_PATTERN.format(root=self.destination_folder, modal=modal, subject=subject,
                                        session=session)
        return p

    def write_output_parquet(self, data: pl.DataFrame, modal: str, subject: any, session: any) -> None:
        """
        Write a processed DataFrame

        Args:
            data: a DF
            modal: modality name (e.g. accelerometer, skeleton, label)
            subject: subject name/ID
            session: session ID
        """
        output_path = self.get_output_file_path(modal=modal, subject=subject, session=session)
        os.makedirs(os.path.split(output_path)[0], exist_ok=True)
        data.write_parquet(output_path)
        logger.info(f'Parquet written: {output_path}')

    def run(self):
        """
        Main processing method
        """
        raise NotImplementedError()


class NpyWindowFormatter:
    def __init__(self, parquet_root_dir: str,
                 window_size_sec: float, step_size_sec: float, min_step_size_sec: float,
                 max_short_window: int = 3,
                 modal_cols: dict = None):
        """
        This class takes result of `ParquetDatasetFormatter`, run sliding window and return as numpy array.

        Args:
            parquet_root_dir: path to processed parquet root dir
            window_size_sec: window size in second
            step_size_sec: step size in second
            min_step_size_sec: for short activity sessions, this is used as minimum step size (in shifting window)
            max_short_window: max number of window for short activity sessions (use shifting window)
            modal_cols: a 2-level dict;
                1st level key is parquet modal name (match with modal name in parquet file path),
                2nd level key is sub-modal name inside the parquet files (any new name),
                2nd level value is a list of column names of that sub-modal. if None, use all columns.
                Example
                    {
                        'inertia': {
                            'acc': ['acc_x', 'acc_y', 'acc_z'],
                            'gyro': ['gyro_x', 'gyro_y', 'gyro_z'],
                        },
                        'skeleton': {
                            'skeleton': None
                        }
                    }
                if `modal_cols` is None (default),
                sub-modal will be the same as modal in parquet path, and all columns are used
        """
        self.parquet_root_dir = parquet_root_dir
        self.window_size_sec = window_size_sec
        self.step_size_sec = step_size_sec
        self.min_step_size_sec = min_step_size_sec
        self.max_short_window = max_short_window
        self.modal_cols = modal_cols

    def get_parquet_files(self) -> pl.DataFrame:
        """
        Scan all parquet files in the root dir

        Returns:
            a DataFrame, each column is a parquet modality, each row is a session, cells are paths to parquet files
        """
        # scan for modal list first
        modal_folders = glob(MODAL_PATH_PATTERN.format(root=self.parquet_root_dir, modal='*'))
        modals = [p.removesuffix('/').split('/')[-1] for p in modal_folders]

        # glob first modal
        first_modal_parquets = sorted(glob(PARQUET_PATH_PATTERN.format(
            root=self.parquet_root_dir,
            modal=modals[0], subject='*', session='*'
        )))
        if len(modals) == 1:
            return pl.DataFrame({modals[0]: first_modal_parquets})

        # check that all modals have the same number of parquet files
        for modal in modals[1:]:
            next_modal_parquets = glob(PARQUET_PATH_PATTERN.format(
                root=self.parquet_root_dir,
                modal=modal, subject='*', session='*'
            ))
            assert len(first_modal_parquets) == len(next_modal_parquets), \
                f'{modals[0]} has {len(first_modal_parquets)} parquet files but {modal} has {len(next_modal_parquets)}'

        # get matching session files of all modals
        result = []
        for first_modal_parquet in first_modal_parquets:
            session_dict = {modals[0]: first_modal_parquet}
            for modal in modals[1:]:
                session_dict[modal] = rreplace(first_modal_parquet, f'/{modals[0]}/', f'/{modal}/')
                assert os.path.isfile(session_dict[modal]), f'Modal parquet file not exist: {session_dict[modal]}'
            result.append(session_dict)

        parquet_files = pl.DataFrame(result)
        return parquet_files

    def get_parquet_session_info(self, parquet_path: str):
        """
        Get session info from parquet file path

        Args:
            parquet_path: parquet file path

        Returns:
            a tuple: (modality, subject id, session id) all elements are string
        """
        info = re.search(PARQUET_PATH_PATTERN.format(
            root=self.parquet_root_dir,
            modal='(.*)', subject='(.*)', session='(.*)'
        ), parquet_path)
        info = tuple(info.group(i) for i in range(1, 4))
        return info

    def run(self) -> pd.DataFrame:
        """
        Main processing method

        Returns:
            a DF, each row is a session, columns are:
                - 'subject': subject ID
                - '<modality 1>': array shape [num window, window length, features]
                - '<modality 2>': ...
                - 'label': array shape [num window]
        """
        raise NotImplementedError()
