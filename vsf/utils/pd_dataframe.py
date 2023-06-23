import os
from typing import Union, Dict

import numpy as np
import pandas as pd
from loguru import logger


def read_df_file(path: str, usecols: list = None, force_column_order: bool = True,
                 **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    This function reads a file into a DataFrame object. Supported formats are: parquet, csv, xls, xlsx.

    Args:
        path: path to file
        usecols: list of columns to read
        force_column_order: column order must be like in `usecols`
        kwargs: keyword arguments for pandas' reading function

    Returns:
        a DataFrame
    """
    if path.endswith('csv'):
        df = pd.read_csv(path, usecols=usecols, **kwargs)
    elif path.endswith('parquet'):
        df = pd.read_parquet(path, columns=usecols, **kwargs)
    elif path.endswith('xlsx') or path.endswith('xls'):
        df = pd.read_excel(path, usecols=usecols, **kwargs)
        if force_column_order and usecols:
            df = df[usecols]
    else:
        raise ValueError('only supports parquet, csv, xls, xlsx')
    return df


def write_df_file(df: pd.DataFrame, path: str, columns: list = None, overwrite: bool = False, **kwargs) -> None:
    """
    Write a DF into a file. Supported formats are: parquet, csv, xlsx

    Args:
        df: Dataframe to write
        path: path to save file
        columns: columns to write, default: all
        overwrite: overwrite if file already exists
        **kwargs: keyword arguments for pandas' writing function
    """
    if (not overwrite) and os.path.isfile(path):
        logger.info(f'Not writing {path} because it already exists.')
        return

    if columns:
        df = df[columns]

    os.makedirs(os.path.split(path)[0], exist_ok=True)

    if path.endswith('csv'):
        df.to_csv(path, index=False, **kwargs)
    elif path.endswith('parquet'):
        df.to_parquet(path, index=False, **kwargs)
    elif path.endswith('xlsx') or path.endswith('xls'):
        df.to_excel(path, index=False, **kwargs)
    else:
        raise ValueError('only supports parquet, csv, xlsx')


def resample_numeric_df(df: pd.DataFrame, timestamp_col: str, new_frequency: float,
                        force_ts_dtype: bool = True) -> pd.DataFrame:
    """
    Resample a DF by linear interpolation.

    Args:
        df: input DF
        timestamp_col: timestamp column name in the DF
        new_frequency: new frequency to interpolate, this must be of the same unit as timestamps;
            example: timestamp unit is millisecond, frequency unit must be sample/millisecond
        force_ts_dtype: convert dtype of new timestamps to the same as old timestamps

    Returns:
        new dataframe
    """

    # get new timestamp array (unit: msec)
    start_ts = df[timestamp_col].iat[0]
    end_ts = df[timestamp_col].iat[-1]
    new_ts = np.arange(np.floor((end_ts - start_ts) * new_frequency + 1)) / new_frequency + start_ts
    if force_ts_dtype:
        new_ts = new_ts.astype(type(start_ts))

    # get data columns
    cols_except_ts = df.columns.to_list()
    cols_except_ts.remove(timestamp_col)
    df_value = df[cols_except_ts].to_numpy()
    df_timestamp = df[timestamp_col].to_numpy()

    # interpolate columns
    new_df = {timestamp_col: new_ts}
    for i, col in enumerate(cols_except_ts):
        new_df[col] = np.interp(x=new_ts, xp=df_timestamp, fp=df_value[:, i])

    new_df = pd.DataFrame(new_df)
    return new_df


def down_sample_df(df: pd.DataFrame, down_sample_by: int) -> pd.DataFrame:
    """
    Down-sample a dataframe by drop rows periodically using numpy.linspace. The first and the last rows will be kept.

    Args:
        df: input dataframe
        down_sample_by: keep 1/N rows of the input DF with N is this parameter

    Returns:
        a smaller DF with the same columns
    """
    if len(df) <= 2:
        logger.info(f'Skip down-sampling because the input DF only have {len(df)} row(s).')
        return df

    num_row = round(len(df) / down_sample_by)
    if num_row == len(df):
        return df

    keep_idx = np.linspace(0, len(df) - 1, num_row, endpoint=True)
    df = df.iloc[keep_idx]
    df = df.reset_index(drop=True)
    return df
