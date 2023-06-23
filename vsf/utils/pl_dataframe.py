import numpy as np
import polars as pl


def resample_numeric_df(df: pl.DataFrame, timestamp_col: str, new_frequency: float,
                        force_ts_dtype: bool = True) -> pl.DataFrame:
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
    start_ts = df.item(0, timestamp_col)
    end_ts = df.item(-1, timestamp_col)
    new_ts = np.arange(np.floor((end_ts - start_ts) * new_frequency + 1)) / new_frequency + start_ts
    if force_ts_dtype:
        new_ts = new_ts.astype(type(start_ts))

    # get data columns
    cols_except_ts = df.columns.copy()
    cols_except_ts.remove(timestamp_col)
    df_value = df.select(cols_except_ts).to_numpy()
    df_timestamp = df.select(timestamp_col).to_numpy().squeeze()

    # interpolate columns
    new_df = {timestamp_col: new_ts}
    for i, col in enumerate(cols_except_ts):
        new_df[col] = np.interp(x=new_ts, xp=df_timestamp, fp=df_value[:, i])

    new_df = pl.DataFrame(new_df)
    return new_df
