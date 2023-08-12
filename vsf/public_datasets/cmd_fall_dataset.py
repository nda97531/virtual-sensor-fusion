import os
import re
from collections import defaultdict
from glob import glob
from typing import List, Dict

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from transforms3d.axangles import axangle2mat

from vsf.public_datasets.base_classes import ParquetDatasetFormatter, NpyWindowFormatter
from vsf.public_datasets.constant import G_TO_MS2
from vsf.utils.number_array import interval_intersection
from vsf.utils.string_utils import rreplace
from vsf.utils.pl_dataframe import resample_numeric_df as pl_resample_numeric_df


class CMDFallConst:
    # modal names
    MODAL_INERTIA = 'inertia'
    MODAL_SKELETON = 'skeleton'

    # raw sampling rates in Hertz
    RAW_INERTIA_FREQ = 50
    RAW_KINECT_FPS = 20

    ACCELEROMETER_POSITION = {1: 'wrist', 155: 'waist'}

    JOINTS_LIST = [
        'hipCenter', 'spine', 'shoulderCenter', 'head',
        'leftShoulder', 'leftElbow', 'leftWrist', 'leftHand',
        'rightShoulder', 'rightElbow', 'rightWrist', 'rightHand',
        'leftHip', 'leftKnee', 'leftAnkle', 'leftFoot',
        'rightHip', 'rightKnee', 'rightAnkle', 'rightFoot'
    ]
    SELECTED_JOINT_LIST = [
        'shoulderCenter',
        'leftElbow', 'leftWrist',
        'rightElbow', 'rightWrist',
        'leftKnee', 'leftAnkle',
        'rightKnee', 'rightAnkle',
    ]

    SELECTED_JOINT_IDX: List[int]
    SELECTED_SKELETON_COLS: List[str]
    SKELETON_ROT_MAT: Dict[int, np.ndarray]

    @classmethod
    def define_att(cls):
        cls.SELECTED_JOINT_IDX = [cls.JOINTS_LIST.index(item) for item in cls.SELECTED_JOINT_LIST]
        cls.SELECTED_SKELETON_COLS = [f'kinect{{kinect_id}}_{joint}_{axis}'
                                      for joint in cls.SELECTED_JOINT_LIST
                                      for axis in ['x', 'y', 'z']]

        # floor equation, mean of all non-zero equations in raw data
        # key: kinect ID, value: equation coefficients [a, b, c, d]
        floor_eqs = {
            3: np.array([0.0277538, 0.9024955, -0.42962335, 1.630657])[[0, 2, 1, 3]]
        }
        cls.SKELETON_ROT_MAT = {}
        for kinect_id, floor_eq in floor_eqs.items():
            norm2 = np.linalg.norm(floor_eq[:3])
            rot_angle = np.arccos(floor_eq[2] / norm2)
            rot_axis = np.cross(floor_eq[:3], [0, 0, 1]) / norm2
            cls.SKELETON_ROT_MAT[kinect_id] = axangle2mat(rot_axis, rot_angle)


CMDFallConst.define_att()


class CMDFallParquet(ParquetDatasetFormatter):
    """
    Class for processing CMDFall dataset.
    Use only Inertial sensors and Camera 3.
    """

    def __init__(self, raw_folder: str, destination_folder: str, sampling_rates: dict,
                 min_length_segment: float = 10,
                 use_accelerometer: list = [155], use_kinect: list = [3]):
        """
        Args:
            min_length_segment: only write segments longer than this threshold (unit: sec)
            use_accelerometer: inertial sensor IDs
            use_kinect: kinect device IDs
        """
        super().__init__(raw_folder, destination_folder, sampling_rates)

        assert len(set(use_accelerometer) - {1, 155}) == 0, 'Invalid inertial sensor ID'
        assert len(set(use_kinect) - set(range(1, 8))) == 0, 'Invalid Kinect ID'

        self.min_length_segment = min_length_segment
        self.use_accelerometer = use_accelerometer
        self.use_kinect = use_kinect

        # if actual interval > expected interval * this coef; it's considered an interruption and DF will be split
        max_interval_coef = 4
        # expected intervals in millisecond
        self.max_interval = {
            CMDFallConst.MODAL_INERTIA: 1000 / CMDFallConst.RAW_INERTIA_FREQ * max_interval_coef,
            CMDFallConst.MODAL_SKELETON: 1000 / CMDFallConst.RAW_KINECT_FPS * max_interval_coef
        }

    @staticmethod
    def get_info_from_session_file(path: str) -> tuple:
        """
        Get info from data file path

        Args:
            path: path to raw accelerometer or skeleton file

        Returns:
            (session ID, subject ID, sensor ID)
        """
        info = re.search(r'S([0-9]*)P([0-9]*)[IK]([0-9]*).txt', path.split('/')[-1])
        info = tuple(int(info.group(i)) for i in range(1, 4))
        return info

    @staticmethod
    def read_accelerometer_df_file(path: str) -> pl.DataFrame:
        """
        Read and format accelerometer file (convert unit, change column names)

        Args:
            path: path to file

        Returns:
            a polars dataframe
        """
        session_id, subject, sensor_id = CMDFallParquet.get_info_from_session_file(path)
        sensor_pos = CMDFallConst.ACCELEROMETER_POSITION[sensor_id]
        df = pl.read_csv(path, columns=['timestamp', 'x', 'y', 'z', 'label'])

        df = df.with_columns(pl.col(['x', 'y', 'z']) * G_TO_MS2)

        df = df.rename({
            'timestamp': 'timestamp(ms)',
            'x': f'{sensor_pos}_acc_x(m/s^2)', 'y': f'{sensor_pos}_acc_y(m/s^2)', 'z': f'{sensor_pos}_acc_z(m/s^2)',
        })
        return df

    @staticmethod
    def normalise_skeletons(skeletons: np.ndarray, kinect_id: int) -> np.ndarray:
        """
        Normalise skeletons

        Args:
            skeletons: 3D array shape [frame, joint, axis]
            kinect_id: kinect ID

        Returns:
            array of the same shape
        """
        # get x,y of hip joint to use as anchor
        anchor_joint_idx = CMDFallConst.JOINTS_LIST.index('hipCenter')
        # remove unused joints
        skeletons = skeletons[:, CMDFallConst.SELECTED_JOINT_IDX + [anchor_joint_idx], :]

        # straighten skeleton (rotate so that it stands up right)
        skeletons = skeletons.transpose([0, 2, 1])
        skeletons = np.matmul(CMDFallConst.SKELETON_ROT_MAT[kinect_id], skeletons)
        skeletons = skeletons.transpose([0, 2, 1])

        # move skeleton to coordinate origin
        anchor_joint_xy = skeletons[:, -1:, :2]
        lowest_z = skeletons[:, :, 2:].min(axis=1, keepdims=True)
        offset = np.concatenate([anchor_joint_xy, lowest_z], axis=2)
        # remove anchor joint
        skeletons = skeletons[:, :-1, :]
        skeletons -= offset

        return skeletons

    @staticmethod
    def read_skeleton_df_file(path: str) -> pl.DataFrame:
        """
        Read and format skeleton file (normalise, change column names)

        Args:
            path: path to file

        Returns:
            a polars dataframe
        """
        session_id, subject, sensor_id = CMDFallParquet.get_info_from_session_file(path)

        df = pl.read_csv(path, skip_rows=1, has_header=False)  # columns=['timestamp', 'x', 'y', 'z', 'label'])
        data_df = df.get_column('column_4')
        info_df = df.select(pl.col('column_1').alias('timestamp(ms)'))
        del df

        # shape [frame, joint * axis]
        data_df = np.array([s.strip().split(' ') for s in data_df], dtype=float)
        org_length = len(data_df)

        # remove 2 RGB columns, keep 3D columns
        # shape [frame, joint, axis]
        data_df = data_df.reshape([org_length, len(CMDFallConst.JOINTS_LIST), 5])
        data_df = data_df[:, :, :3]
        # switch Y and Z
        data_df = data_df[:, :, [0, 2, 1]]
        # normalise skeleton
        data_df = CMDFallParquet.normalise_skeletons(data_df, sensor_id)

        # shape [frame, joint * axis]
        data_df = data_df.reshape([org_length, len(CMDFallConst.SELECTED_JOINT_LIST) * 3])
        data_df = pl.DataFrame(data_df,
                               schema=[c.format(kinect_id=sensor_id) for c in CMDFallConst.SELECTED_SKELETON_COLS])

        df = pl.concat([info_df, data_df], how='horizontal')
        return df

    def process_session(self, data_files: dict) -> list:
        """
        Produce a fully processed dataframe for each modal in a session

        Args:
            data_files: a dict with keys are modal names, values are raw data paths

        Returns:
            list of uninterrupted segments; each one is a dict with keys are modal names, values are DFs
        """
        # read all DFs
        # key: modal; value: whole original session DF
        data_dfs = {
            sensor: self.read_accelerometer_df_file(data_file) if sensor.startswith(
                CMDFallConst.MODAL_INERTIA)
            else self.read_skeleton_df_file(data_file)
            for sensor, data_file in data_files.items()
        }

        # split interrupted signals into sub-sessions
        # key: modal; value: timestamp segments (start & end ts)
        ts_segments = {}
        for sensor, df in data_dfs.items():
            ts = df.get_column('timestamp(ms)').to_numpy()
            intervals = np.diff(ts)
            interruption_idx = np.nonzero(intervals > self.max_interval[sensor.split('_')[0]])[0]
            interruption_idx = np.concatenate([[-1], interruption_idx, [len(intervals)]])
            ts_segments[sensor] = [
                [ts[interruption_idx[i - 1] + 1], ts[interruption_idx[i]]]
                for i in range(1, len(interruption_idx))
            ]
        combined_ts_segments = interval_intersection(list(ts_segments.values()))
        logger.info(f'Number of segments: ' + '; '.join(f'{k}: {len(v)}' for k, v in ts_segments.items()))

        # crop segments based on timestamps found above
        results = []
        label_df: pl.DataFrame = None
        kept_segments = 0
        kept_time = 0
        total_time = 0
        # for each segment
        for combined_ts_segment in combined_ts_segments:
            segment_length = (combined_ts_segment[1] - combined_ts_segment[0]) / 1000
            total_time += segment_length
            if segment_length < self.min_length_segment:
                continue
            kept_time += segment_length
            kept_segments += 1

            # dict key: sensor type; value: concatenated DF of all sensors of this type
            segment_dfs = defaultdict(list)
            # for each sensor
            for sensor, df in data_dfs.items():
                # remove sensor ID, keep sensor type
                sensor = sensor.split('_')[0]

                # crop the segment in sensor DF
                df = df.filter(
                    (pl.col('timestamp(ms)') >= combined_ts_segment[0]) &
                    (pl.col('timestamp(ms)') <= combined_ts_segment[1])
                )
                # get label DF if not already exists
                if (sensor == CMDFallConst.MODAL_INERTIA) and (label_df is None):
                    label_df = df.select('timestamp(ms)', 'label')
                    if label_df.get_column('timestamp(ms)').is_sorted():
                        label_df = label_df.set_sorted('timestamp(ms)')
                    else:
                        label_df = label_df.sort(by='timestamp(ms)')
                        logger.info('Sorting because not already sorted')

                # interpolate to resample
                df = df.select(pl.exclude('label'))
                df = pl_resample_numeric_df(df, 'timestamp(ms)', self.sampling_rates[sensor],
                                            start_ts=combined_ts_segment[0], end_ts=combined_ts_segment[1])
                # only keep ts column for 1 DF of each sensor for later concatenation
                if len(segment_dfs[sensor]):
                    df = df.drop('timestamp(ms)')
                segment_dfs[sensor].append(df)

            # concat DFs with the same sensor type, add label column
            for sensor in segment_dfs.keys():
                this_df = pl.concat(segment_dfs[sensor], how='horizontal')

                # assign labels
                # it is sorted after interpolation
                this_df = this_df.set_sorted('timestamp(ms)')
                this_df = this_df.join_asof(label_df, on='timestamp(ms)', strategy='nearest')
                segment_dfs[sensor] = this_df

            results.append(segment_dfs)
        logger.info(f'Kept {kept_segments}/{len(combined_ts_segments)} segment(s)')
        logger.info('Kept %.02f/%.02f (sec); %.02f%%' % (kept_time, total_time, kept_time / total_time * 100))
        return results

    def run(self):
        assert len(self.use_accelerometer), 'No accelerometer is used?'

        logger.info('Scanning for sessions...')

        # scan inertial sensor files
        first_inertia_name = f'{CMDFallConst.MODAL_INERTIA}_{self.use_accelerometer[0]}'
        session_files = {
            first_inertia_name: sorted(
                glob(f'{self.raw_folder}/accelerometer/*I{self.use_accelerometer[0]}.txt'))
        }
        old_path_tail = f'I{self.use_accelerometer[0]}.txt'
        for inertia_sensor_id in self.use_accelerometer[1:]:
            new_path_tail = f'I{inertia_sensor_id}.txt'
            session_files[f'{CMDFallConst.MODAL_INERTIA}_{inertia_sensor_id}'] = [
                rreplace(path, old_path_tail, new_path_tail)
                for path in session_files[first_inertia_name]
            ]

        # scan skeleton files
        for kinect_id in self.use_kinect:
            new_path_tail = f'K{kinect_id}.txt'
            session_files[f'{CMDFallConst.MODAL_SKELETON}_{kinect_id}'] = [
                rreplace(rreplace(path, old_path_tail, new_path_tail), 'accelerometer', 'skeleton')
                for path in session_files[first_inertia_name]
            ]

        session_files = pd.DataFrame(session_files)
        logger.info(f'Found {len(session_files)} sessions in total')

        skip_session_files = 0
        write_segment_files = 0
        # for each session
        for _, session_row in session_files.iterrows():
            # get session info
            session_id, subject, sensor_id = self.get_info_from_session_file(session_row.iat[0])
            session_info = f'S{session_id}P{subject}'

            if os.path.isfile(self.get_output_file_path(CMDFallConst.MODAL_INERTIA, subject, session_info)) \
                    and os.path.isfile(self.get_output_file_path(CMDFallConst.MODAL_SKELETON, subject, session_info)):
                logger.info(f'Skipping session {session_info} because already run before')
                skip_session_files += 1
                continue
            logger.info(f'Starting session {session_info}')

            # get data
            data_segments = self.process_session(session_row.to_dict())
            for i, segment in enumerate(data_segments):
                this_session_info = f'{session_info}_{i}' if i < len(data_segments) - 1 else session_info
                inertial_df = segment[CMDFallConst.MODAL_INERTIA]
                skeleton_df = segment[CMDFallConst.MODAL_SKELETON]
                # write files
                if self.write_output_parquet(inertial_df, CMDFallConst.MODAL_INERTIA, subject, this_session_info):
                    write_segment_files += 1
                if self.write_output_parquet(skeleton_df, CMDFallConst.MODAL_SKELETON, subject, this_session_info):
                    write_segment_files += 1
        logger.info(f'{write_segment_files} file(s) written, {skip_session_files} session(s) skipped')


class CMDFallNpyWindow(NpyWindowFormatter):
    pass
    # def run(self) -> pd.DataFrame:
    #     # get list of parquet files
    #     parquet_sessions = self.get_parquet_file_list()
    #
    #     result = []
    #     # for each session
    #     for parquet_session in parquet_sessions.iter_rows(named=True):
    #         # get session info
    #         _, subject, session_id = self.get_parquet_session_info(list(parquet_session.values())[0])
    #         session_regex = re.match('Subject(?:[0-9]*)Activity([0-9]*)Trial([0-9]*)', session_id)
    #         session_label, session_trial = (int(session_regex.group(i)) for i in [1, 2])
    #
    #         session_result = self.process_parquet_to_windows(
    #             parquet_session=parquet_session,
    #             subject=subject,
    #             session_label=session_label,
    #             is_short_activity=session_label in CMDFallConst.SHORT_ACTIVITIES
    #         )
    #
    #         # add trial info
    #         session_result['trial'] = session_trial
    #
    #         result.append(session_result)
    #     result = pd.DataFrame(result)
    #     return result


if __name__ == '__main__':
    parquet_dir = '/mnt/data_drive/projects/UCD04 - Virtual sensor fusion/processed_parquet/CMDFall'
    inertial_freq = 50
    skeletal_freq = 20
    window_size_sec = 4
    step_size_sec = 1.5
    min_step_size_sec = 0.5

    CMDFallParquet(
        raw_folder='/mnt/data_drive/projects/raw datasets/CMDFall',
        destination_folder=parquet_dir,
        sampling_rates={CMDFallConst.MODAL_INERTIA: inertial_freq,
                        CMDFallConst.MODAL_SKELETON: skeletal_freq},
        use_accelerometer=[1, 155],
        use_kinect=[3]
    ).run()

    # CMDFall = CMDFallNpyWindow(
    #     parquet_root_dir=parquet_dir,
    #     window_size_sec=window_size_sec,
    #     step_size_sec=step_size_sec,
    #     min_step_size_sec=min_step_size_sec,
    #     max_short_window=3,
    #     modal_cols={
    #         CMDFallConst.MODAL_INERTIA: {
    #             'wrist_acc': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)'],
    #             'belt_acc': ['belt_acc_x(m/s^2)', 'belt_acc_y(m/s^2)', 'belt_acc_z(m/s^2)']
    #         },
    #         CMDFallConst.MODAL_SKELETON: {
    #             'skeleton': None
    #         }
    #     }
    # ).run()
