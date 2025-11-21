import os
import random
import re

import mne
import numpy as np
from tqdm import tqdm
import scipy
import scipy.io as scio
from scipy import signal
import pickle
import pandas as pd
from collections import OrderedDict
from pathlib import Path
import math
from typing import Tuple, List, Optional
import glob
import pyprep

from clip_extraction_utils import ContinuousBlockLabelProcessor

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

setup_seed(1)

class utils():
    @staticmethod
    def unit_change(raw_file, start_samp, stop_samp, units='uV'): # raw_file is originally raw_cnt
        #data = raw_file.get_data(start=start_samp, stop=stop_samp, units=units) # is it ok even if the file type is not cnt
        # TODO : 이거 그냥 수동 값 scaling으로 갈아끼우고 get_data에서 crop을 좀 더 이쁘게 할 수 있었으면 좋겠어요
        
        data = None # use 'if' -> make each point

        if hasattr(raw_file, 'get_data'):
            data = raw_file.get_data(start=start_samp, stop=stop_samp, units=units)

        elif hasattr(raw_file, 'data') and hasattr(raw_file, 'unit'): # 승주쌤께 method 어떻게 사용하셨는지 여쭤보기
            raw_data = raw_file.data[start_samp:stop_samp]

            unit_in_file = raw_file.unit.lower()
            if unit_in_file in ['volt', 'volts']:
                scale = 1e6  # → µV
            elif unit_in_file in ['millivolt', 'millivolts']:
                scale = 1e3
            elif unit_in_file in ['microvolt', 'microvolts']:
                scale = 1
            else:
                raise ValueError(f"Unknown unit in file: {unit_in_file}")

            if units == 'uV':
                target_scale = 1
            elif units == 'mV':
                target_scale = 1e-3
            elif units == 'V':
                target_scale = 1e-6
            else:
                raise ValueError(f"Unsupported output unit: {units}")

            data = raw_data[:] * (scale * target_scale)

            # Transpose if needed: (time, channels) → (channels, time)
            if data.shape[0] > data.shape[1]:
                data = data.T

        else:
            raise TypeError("Unsupported raw_file type. Must be MNE Raw or pynwb ElectricalSeries.")

        return np.array(data)
        
        return data # 아예 처음에 이렇게 받아와서 메모리에 올려야 함 / np.array로 저장
    # segment chunk 정보를 나중에 받아오게 되니 그 segment 덩어리를 list 등으로 받아서 처리할 수 있도록

    @staticmethod
    def bad_channel_delete(raw_cnt, bad_channel_list=['EKG1','EKG2','Cz']):
        # TODO : 이거 성진쌤께서 주신 masks랑 연동 잘 되도록 고쳐주세요
        """
        delete bad channel
        cnt_path: file path
        CHUNK_SEC: segment length
        bad_channel_list: channel list to delete
        """
        # set values for func
        #raw_cnt   = mne.io.read_raw_cnt(cnt_path, preload=False, verbose=False) # cnt 파일이 아닌 경우 생각하기
        srate     = raw_cnt.info['sfreq']
        n_times   = raw_cnt.n_times
        total_sec = n_times / srate
        #n_chunks  = math.ceil(total_sec / CHUNK_SEC)

        # delete bad channel
        drop_chs = [c for c in bad_channel_list if c in raw_cnt.ch_names]
        if drop_chs:
            raw_cnt.drop_channels(drop_chs)
        #ch_names = raw_cnt.ch_names
        #n_ch     = len(ch_names)

        return raw_cnt
    
    @staticmethod
    def _get_csv_path(file_path: str):
        base_dir = os.path.dirname(file_path)
        subject_name = os.path.basename(file_path).split('_')[0]
        csv_path = os.path.join(base_dir, f"{subject_name}.csv")
        return csv_path

    @staticmethod
    def _load_xyz_from_csv(file_path: str,
                           want_channels) -> np.ndarray:
        # TODO : 이거 성진쌤께서 주신 csv 파일들이 형식 조금씩 다른 경우(항목 이름이 MNI_X와 같은 식으로 되어서 MNI와 Talairach 구분) 있는 것 같아요 주의해주세요
        """
        csv 안에서 xyz 좌표 읽어서 array로 저장
        """
        csv_path = utils._get_csv_path(file_path)
        df = pd.read_csv(csv_path, delim_whitespace=True)

        xyz_array = df[['X', 'Y', 'Z']].to_numpy()

        return xyz_array

    @staticmethod
    def _load_xyz_from_csv_peers(csv_path: str,
                           want_channels=None) -> np.ndarray:
        """
        csv 안에서 xyz 좌표 읽어서 array로 저장
        """
        # 공백으로 구분된 파일 읽기
        #df = pd.read_csv(csv_path, delim_whitespace=True)
        df = pd.read_csv(csv_path)

        xyz_array = df[['x_mm', 'y_mm', 'z_mm']].to_numpy()

        return xyz_array

    @staticmethod
    def _load_xyz_meta_data_from_csv(file_path: str,
                                    want_channels):
        """
        csv 안에서 xyz의 meta data (type, group, depthlabel, bad + (channel?))
        """
        csv_path = utils._get_csv_path(file_path)
        df = pd.read_csv(csv_path, delim_whitespace=True)

        channel_list     = df['Channel'].tolist()
        type_list        = df['Type'].tolist()
        group_list       = df['Group'].tolist()
        depthlabel_list  = df['DepthLabel'].tolist()
        bad_list         = df['Bad'].tolist()
        return channel_list, type_list, group_list, depthlabel_list, bad_list

    @staticmethod
    def _load_xyz_from_sfr(sfp_path: str,
                           want_channels=None) -> np.ndarray:
        "load MNI xyz position from sfr file(GSN129)"
        pos_dict = OrderedDict()
        with open(sfp_path, 'r') as f:
            lines = f.readlines()[3:]
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    ch_name = parts[0].upper()
                    xyz = np.array([float(p) * 10 for p in parts[1:4]])
                    pos_dict[ch_name] = xyz

        xyz_array = np.array(list(pos_dict.values()))
        return xyz_array

    @staticmethod
    def _load_xyz_from_elc(elc_path: str,
                        want_channels: list[str]) -> np.ndarray:
        "load MNI xyz position from elc file(1005)"
        want_up = [ch.upper() for ch in want_channels]

        # start reading
        with open(elc_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()[4:]

        positions_start = labels_start = None
        for i, ln in enumerate(lines):
            ll = ln.strip().lower()
            if ll == "positions":
                positions_start = i + 1
            elif ll == "labels":
                labels_start = i + 1
                break
        if positions_start is None or labels_start is None:
            print(f"DEBUG: ELC file: {elc_path}")
            print(f"DEBUG: First 10 lines after header: {[ln.strip() for ln in lines[:10]]}")
            print(f"DEBUG: positions_start={positions_start}, labels_start={labels_start}")
            raise RuntimeError(f"No Position/Labels section in elc file. positions_start={positions_start}, labels_start={labels_start}")

        # read position
        positions = []
        for ln in lines[positions_start:labels_start-1]:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            xyz = [float(p) for p in ln.split()[:3]]
            positions.append(np.array(xyz, dtype=float))

        # read electrode name
        labels = [ln.strip().upper() for ln in lines[labels_start:]
                if ln.strip() and not ln.startswith('#')]

        if len(labels) != len(positions):
            raise RuntimeError("The number of positions and that of labels are different.")

        # mapping electrode names and positions (keep order)
        xyz_list = []
        for ch in want_up:
            if ch in labels:
                idx = labels.index(ch)
                xyz_list.append(positions[idx])
            else:
                print(f"[ELC] Warning: {ch} not found; NaN inserted")
                xyz_list.append(np.full(3, np.nan))
        return np.vstack(xyz_list)          # shape = (len(want), 3)

    @staticmethod
    def _load_xyz_from_mat(file_path: str, want_channels) -> np.ndarray:
        "load xyz position from locs mat file"
        channels = want_channels(file_path)
        channel_idx = [int(ch) for ch in channels]

        file = Path(file_path)
        subj = file.parent.name
        # p.parents[2] is the “faces_basic” folder, so we go up two levels and then into "locs"
        locs_path = file.parents[2] / "locs" / f"{subj}_xslocs.mat"
        
        # Check if the locs file exists
        if not locs_path.exists():
            raise FileNotFoundError(f"Locs file not found: {locs_path}")
        
        # Load the locs file
        locs_file = scio.loadmat(locs_path)

        # Extract coordinates for the specified channels
        coords = locs_file['locs'][channel_idx, :]   # shape (len(channel_idx), 3)
        
        return np.asarray(coords, dtype=float)
    
    def filter_ecog_data(multichannel_signal: np.ndarray, fs=1000, powerline_freq=60, Qfactor=35):
        """
        Harmonics removal and frequency filtering using scipy's iirnotch.
        
        Parameters:
            multichannel_signal (np.ndarray) : Multi-channel signal to be filtered
            fs (int) : Sampling rate of the signal
            powerline_freq (int) : Grid Frequency
            Qfactor (float) : Quality factor for the notch filter, higher values mean narrower notch

        Returns:
            np.ndarray : Filtered signal with powerline noise removed
        """
        # Q factor에 대한 맞춤 적용 코드 및 Q factor 조정 확인하기
        print("Starting Notch filtering...")

        multichannel_signal = multichannel_signal.astype(float)

        harmonics = np.array([i * powerline_freq for i in range(1, (fs // 2) // powerline_freq)])
        
        # Notch filter design
        for freq in harmonics:
            wo = freq / (fs / 2)  # Normalized frequency
            bw = wo / Qfactor
            b, a = scipy.signal.iirnotch(wo, bw)

            # Apply notch filter to each channel
            filtered_signal = np.zeros_like(multichannel_signal)
            for i in range(multichannel_signal.shape[1]):  # Iterate over channels
                filtered_signal[:, i] = scipy.signal.filtfilt(b, a, multichannel_signal[:, i])

        print("Powerline noise removed...")
        return filtered_signal


class DefaultDatasetSetting():
    def __init__(self):
        self.name = 'default'

    @staticmethod
    def file_list():
        raise NotImplementedError("file_list must be implemented in subclass")

    @staticmethod
    def filter():
        raise NotImplementedError("file_list must be implemented in subclass")
    
    @staticmethod
    # default shaping
    def shaping(raw, original_sampling_rate, segment_len):
        array = raw.to_data_frame().values[:, 1:]
        points, chs = array.shape
        if points < segment_len * 10 * original_sampling_rate:
            return None, chs
        trim = points % (segment_len * original_sampling_rate)
        array = array[2*segment_len*original_sampling_rate:-(trim+2*segment_len*original_sampling_rate), :]
        array = array.reshape(-1, segment_len, original_sampling_rate, chs).transpose(0, 3, 1, 2)  # (N,C,T,D)
        return array, chs
    
    @staticmethod
    # default qc
    def qc(sample):
        return np.max(np.abs(sample)) < 100
    
    @staticmethod
    # default resample
    def resample(sample, chs, original_sampling_rate, target_sampling_rate, segment_len):
        if original_sampling_rate == target_sampling_rate:
            sample_rs = sample
        else: 
            sample_2d  = sample.reshape(chs, -1)
            sample_rs = mne.filter.resample(
                        sample_2d,
                        up=target_sampling_rate,
                        down=original_sampling_rate,
                        npad='auto',
                        axis=1
                    )
            sample_rs  = sample_rs.reshape(chs, segment_len, target_sampling_rate)
        return sample_rs
    
    @staticmethod
    def set_meta_data():
        raise NotImplementedError("set_meta_data must be implemented in subclass")

    @staticmethod
    # default label(pretrain)
    def label(raw):
        return None

class TUEGDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'TUEG'

    @staticmethod
    def file_list(rootDir, ext_filter, percent):
        file_path_list = []
        walker = tqdm(os.walk(rootDir), desc="Walking through directories", mininterval=10)

        for root, dirs, files in walker:
            for file in files:
                if ext_filter:
                    if isinstance(ext_filter, str):
                        if not file.endswith(ext_filter):
                            continue
                    elif isinstance(ext_filter, (list, tuple)):
                        if not any(file.endswith(ext) for ext in ext_filter):
                            continue
                file_path_list.append(os.path.join(root, file))
        file_path_list = sorted(file_path_list)
        #random.shuffle(file_path_list)
        
        if percent == 1.0:
            samples_list_to_process = file_path_list[:int(len(file_path_list) * 0.25)]
        elif percent == 2.0:
            samples_list_to_process = file_path_list[int(len(file_path_list) * 0.25):int(len(file_path_list) * 0.5)]
        elif percent == 3.0:
            samples_list_to_process = file_path_list[int(len(file_path_list) * 0.5):int(len(file_path_list) * 0.75)]
        elif percent == 4.0:
            samples_list_to_process = file_path_list[int(len(file_path_list) * 0.75):]
            
        #samples_list_to_process = file_path_list[:int(len(file_path_list) * percent)] # for default
        #samples_list_to_process = file_path_list[:431] # for 10,000 raw key
        #samples_list_to_process = file_path_list[:1] # for minimal sample
        return {'all': samples_list_to_process}

    @staticmethod
    def qc(
        sample: np.ndarray,
        xyz: np.ndarray,
        ch_names: List[str], 
        threshold: float = 100.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        채널별 QC → bad channel 삭제 → segment 전체 폐기 or clipping 후 반환
        Inputs:
        sample   : np.ndarray, shape (n_ch, n_sec, fs)
        xyz      : np.ndarray, shape (n_ch, 3)
        ch_names : List[str], length n_ch
        threshold: float
        Returns:
        (cleaned_sample, cleaned_xyz, cleaned_ch_names)
        or (None, None, None) if 전체 segment 폐기
        """
        n_ch, n_sec, fs = sample.shape
        per_ch_size = n_sec * fs        

        # 1) threshold 초과 마스크
        abs_sample = np.abs(sample)
        over_mask = abs_sample > threshold
        num_over = over_mask.sum(axis=(1, 2))             # 채널별 초과 개수

        # 2) bad channel 판정
        bad_ch_mask = num_over >= (per_ch_size *0.0333)       # True = bad
        n_bad = bad_ch_mask.sum()

        # 3) 채널 절반 이상 bad → segment 폐기
        if n_bad >= (n_ch *0.5):
            return None, None, None

        # 4) 남은(“good”) 채널 인덱스
        good_idx = np.nonzero(~bad_ch_mask)[0]

        # 5) cleaned sample 생성 + clipping
        cleaned = sample[good_idx].copy().astype(np.float32)
        cleaned_over = over_mask[good_idx]
        cleaned[cleaned_over] = np.sign(cleaned[cleaned_over]) * threshold

        # 6) xyz, ch_names 필터링
        cleaned_xyz   = xyz[good_idx]
        cleaned_names = [ch_names[i] for i in good_idx]

        return cleaned, cleaned_xyz, cleaned_names

    @staticmethod
    def filter(raw, highpass, lowpass, notch, notch_auto=True):
        original_sampling_rate = int(raw.info["sfreq"])
        if lowpass is None:
            raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        if notch_auto:
            nyquist = original_sampling_rate / 2 
            harmonics = []
            for base in notch:
                mult = 1 
                while base * mult <= nyquist:
                    harmonics.append(base * mult)
                    mult += 1
            harmonics = sorted(set(harmonics))
            raw.notch_filter(harmonics)
        else:
            raw.notch_filter(notch)
        return raw, original_sampling_rate

    @staticmethod
    def set_meta_data(file_path, seg_idx):
    #def set_meta_data_TUEG(file_path, file_key_list, seg_idx):
        file_name = file_path.split('/')[-1][:-4]
        sample_key = f'TUEG-{file_name}_{seg_idx}'
        #file_key_list.append(sample_key)
        subject_id = file_name.split('_')[0] # set subject_id in TUEG
        return sample_key, subject_id, None, None

########################################################################
# ADFTD Dataset Setting - Added for OpenNeuro ds004504 dataset
# Author: Claude Code
# Date: 2025-10-17
# Description: Dataset setting for ADFTD (Alzheimer's Disease and
#              Frontotemporal Dementia) EEG dataset from OpenNeuro
#              Dataset: https://openneuro.org/datasets/ds004504/versions/1.0.8
#              Paper: "A Dataset of Scalp EEG Recordings of Alzheimer's
#                      Disease, Frontotemporal Dementia and Healthy Subjects"
########################################################################
class ADFTDDatasetSetting(DefaultDatasetSetting):
    # Class-level cache for label mapping
    _label_cache = None
    _participants_path = None

    def __init__(self):
        super().__init__()
        self.name = 'ADFTD'

    @staticmethod
    def file_list(root_dir: str, split: dict):
        """
        Get list of .set files from ADFTD dataset with stratified random split
        - Binary classification: HC vs AD only (FTD excluded)
        - Stratified random split with seed=42 (as per LEAD paper)
        - Each group (HC, AD) split at 60:20:20 ratio
        - BIDS structure: sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set

        Args:
            root_dir: Root directory containing subject folders
            split: Dict with train/val/test slices (ignored, split done here)

        Returns:
            Dict with keys 'train', 'val', 'test' containing file paths
        """
        import pandas as pd

        # Load participants.tsv to get group labels
        participants_path = os.path.join(root_dir, 'participants.tsv')
        if not os.path.exists(participants_path):
            participants_path = os.path.join(root_dir, 'ds004504', 'participants.tsv')

        if not os.path.exists(participants_path):
            raise FileNotFoundError(f"participants.tsv not found in {root_dir}")

        participants = pd.read_csv(participants_path, sep='\t')
        print(f"[ADFTD] Loaded participants.tsv with {len(participants)} subjects")

        # Collect all .set files
        file_path_list = []
        walker = tqdm(os.walk(root_dir), desc="Walking through ADFTD directories", mininterval=10)

        for root, dirs, files in walker:
            for file in files:
                if not file.endswith('.set'):
                    continue
                if 'derivatives' in root:
                    continue
                file_path_list.append(os.path.join(root, file))

        file_path_list = sorted(file_path_list)
        print(f"[ADFTD] Found {len(file_path_list)} .set files")

        # Create mapping: subject_id -> (file_path, group)
        subject_files = {}
        for file_path in file_path_list:
            match = re.search(r'sub-(\d+)', file_path)
            if match:
                sub_id = match.group(1)
                participant_id = f'sub-{sub_id}'
                row = participants[participants['participant_id'] == participant_id]
                if not row.empty:
                    group = row['Group'].values[0]
                    subject_files[sub_id] = (file_path, group)

        # Separate HC and AD subjects (exclude FTD)
        hc_files = [(sid, fpath) for sid, (fpath, group) in subject_files.items() if group == 'C']
        ad_files = [(sid, fpath) for sid, (fpath, group) in subject_files.items() if group == 'A']
        ftd_files = [(sid, fpath) for sid, (fpath, group) in subject_files.items() if group == 'F']

        print(f"[ADFTD] Subject counts by group:")
        print(f"  - HC (C): {len(hc_files)} subjects")
        print(f"  - AD (A): {len(ad_files)} subjects")
        print(f"  - FTD (F): {len(ftd_files)} subjects (EXCLUDED from training)")

        # Stratified random split with seed=42
        random.seed(42)
        random.shuffle(hc_files)
        random.shuffle(ad_files)

        # Split each group at 60:20:20 ratio
        hc_train = [f[1] for f in hc_files[:int(0.6 * len(hc_files))]]
        hc_val = [f[1] for f in hc_files[int(0.6 * len(hc_files)):int(0.8 * len(hc_files))]]
        hc_test = [f[1] for f in hc_files[int(0.8 * len(hc_files)):]]

        ad_train = [f[1] for f in ad_files[:int(0.6 * len(ad_files))]]
        ad_val = [f[1] for f in ad_files[int(0.6 * len(ad_files)):int(0.8 * len(ad_files))]]
        ad_test = [f[1] for f in ad_files[int(0.8 * len(ad_files)):]]

        # Combine HC and AD for each split
        parts = {
            'train': hc_train + ad_train,
            'val': hc_val + ad_val,
            'test': hc_test + ad_test
        }

        print(f"[ADFTD] Stratified split (seed=42):")
        print(f"  - train: {len(parts['train'])} subjects (HC:{len(hc_train)}, AD:{len(ad_train)})")
        print(f"  - val:   {len(parts['val'])} subjects (HC:{len(hc_val)}, AD:{len(ad_val)})")
        print(f"  - test:  {len(parts['test'])} subjects (HC:{len(hc_test)}, AD:{len(ad_test)})")

        return parts

    @staticmethod
    def filter(raw, highpass, lowpass, notch, notch_auto=True):
        """
        Filter function for ADFTD dataset
        - Original sampling rate: 500 Hz
        - Recommended: 0.5-45 Hz bandpass (as per dataset preprocessing)
        """
        original_sampling_rate = int(raw.info["sfreq"])
        if lowpass is None:
            raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        if notch_auto:
            nyquist = original_sampling_rate / 2
            harmonics = []
            for base in notch:
                mult = 1
                while base * mult <= nyquist:
                    harmonics.append(base * mult)
                    mult += 1
            harmonics = sorted(set(harmonics))
            raw.notch_filter(harmonics)
        else:
            raw.notch_filter(notch)
        return raw, original_sampling_rate

    @staticmethod
    def shaping(raw, original_sampling_rate, segment_len, version='v1'):
        """
        Two versions of segmentation for ADFTD dataset

        Version 1 (v1): Single-scale 30-second segments at 500Hz
        - Output: (19, 30, 500)
        - For standard EEG analysis

        Version 2 (v2): Multi-scale 1s/2s/4s segments at 500/250/125Hz
        - Output: (19, 1, 500), (19, 2, 250), (19, 4, 125)
        - For multi-resolution analysis (LEAD-style)
        - 50% overlap for each scale

        Args:
            raw: MNE Raw object
            original_sampling_rate: Original sampling rate (500Hz)
            segment_len: Segment length in seconds (30 for v1, ignored for v2)
            version: 'v1' or 'v2' (default: 'v1')

        Returns:
            For v1: array (N, 19, 30, 500), int (19)
            For v2: list of dicts with 'data' and 'marker', int (19)
        """
        if version == 'v1':
            return ADFTDDatasetSetting._shaping_v1(raw, original_sampling_rate, segment_len)
        elif version == 'v2':
            return ADFTDDatasetSetting._shaping_v2(raw, original_sampling_rate)
        else:
            raise ValueError(f"Unknown version: {version}. Use 'v1' or 'v2'")

    @staticmethod
    def _shaping_v1(raw, original_sampling_rate, segment_len=30):
        """
        Version 1: Single-scale 30-second segments at 500Hz
        Output: (N, 19, 30, 500)
        """
        # Get data and convert to μV
        array = raw.to_data_frame(scalings={'eeg': 1e6}).values[:, 1:]
        points, chs = array.shape

        # Check minimum length requirement
        if points < segment_len * 10 * original_sampling_rate:
            return None, chs

        # Channel mapping
        rename_mapping = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        raw.rename_channels(rename_mapping)
        target_19_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                              'T7', 'C3', 'Cz', 'C4', 'T8',
                              'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
        raw.pick_channels(target_19_channels, ordered=True)
        array_19 = raw.to_data_frame(scalings={'eeg': 1e6}).values[:, 1:]

        # Maintain 500Hz
        if original_sampling_rate != 500:
            array_500 = mne.filter.resample(array_19.T, up=500, down=original_sampling_rate).T
        else:
            array_500 = array_19

        # Segment into 30-second windows without overlap
        window = segment_len * 500
        total_points = len(array_500)
        num_segments = total_points // window
        trim_points = total_points - (num_segments * window)
        trim_start = trim_points // 2
        trim_end = trim_points - trim_start

        if trim_end > 0:
            data_trimmed = array_500[trim_start:-trim_end, :]
        else:
            data_trimmed = array_500[trim_start:, :]

        array_segments = []
        for i in range(num_segments):
            start = i * window
            end = start + window
            seg = data_trimmed[start:end, :]  # (15000, 19)
            seg_shaped = seg.T.reshape(19, segment_len, 500)  # (C, T_sec, D_hz)
            array_segments.append(seg_shaped)

        array = np.stack(array_segments, axis=0).transpose(0, 1, 2, 3)  # (N, 19, 30, 500)
        return array, 19

    @staticmethod
    def _shaping_v2(raw, original_sampling_rate):
        """
        Version 2: Multi-scale 1s/2s/4s segments at 500/250/125Hz
        Output: list of dicts with shapes (19, 1, 500), (19, 2, 250), (19, 4, 125)
        50% overlap for each scale
        """
        # Get data and convert to μV
        array = raw.to_data_frame(scalings={'eeg': 1e6}).values[:, 1:]
        points, chs = array.shape

        # Check minimum length requirement (at least 10 seconds)
        if points < 10 * original_sampling_rate:
            return None, chs

        # Channel mapping
        rename_mapping = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        raw.rename_channels(rename_mapping)
        target_19_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                              'T7', 'C3', 'Cz', 'C4', 'T8',
                              'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
        raw.pick_channels(target_19_channels, ordered=True)
        array_19 = raw.to_data_frame(scalings={'eeg': 1e6}).values[:, 1:]

        # Maintain 500Hz base
        if original_sampling_rate != 500:
            array_500 = mne.filter.resample(array_19.T, up=500, down=original_sampling_rate).T
        else:
            array_500 = array_19

        all_segments = []

        # 500Hz: 1sec windows (50% overlap)
        data_500 = array_500
        window_500, step_500 = 500, 250
        for start in range(0, len(data_500) - window_500 + 1, step_500):
            seg = data_500[start:start+window_500, :]  # (500, 19)
            seg_lmdb = seg.T.reshape(19, 1, 500)  # (C, T_sec, D_hz)
            all_segments.append({'data': seg_lmdb, 'marker': 500})

        # 250Hz: 2sec windows (50% overlap)
        data_250 = mne.filter.resample(array_500.T, up=1, down=2).T
        window_250, step_250 = 500, 250
        for start in range(0, len(data_250) - window_250 + 1, step_250):
            seg = data_250[start:start+window_250, :]  # (500, 19)
            seg_lmdb = seg.T.reshape(19, 2, 250)  # (C, T_sec, D_hz)
            all_segments.append({'data': seg_lmdb, 'marker': 250})

        # 125Hz: 4sec windows (50% overlap)
        data_125 = mne.filter.resample(array_500.T, up=1, down=4).T
        window_125, step_125 = 500, 250
        for start in range(0, len(data_125) - window_125 + 1, step_125):
            seg = data_125[start:start+window_125, :]  # (500, 19)
            seg_lmdb = seg.T.reshape(19, 4, 125)  # (C, T_sec, D_hz)
            all_segments.append({'data': seg_lmdb, 'marker': 125})

        return all_segments, 19

    @staticmethod
    def set_meta_data(file_path, seg_idx):
        """
        Extract metadata from BIDS-formatted file path
        - Subject ID from sub-XXX folder or filename
        - Task: typically 'eyesclosed' for this dataset
        """
        # Extract subject info from BIDS path: sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
        parts = file_path.replace('\\', '/').split('/')
        subject_id = None
        task = 'eyesclosed'  # Default task for ADFTD dataset

        # Try to extract from path
        for part in parts:
            if part.startswith('sub-'):
                subject_id = part.replace('sub-', '')
                break

        # Fallback: extract from filename
        if subject_id is None:
            file_name = os.path.basename(file_path)
            if '_task-' in file_name:
                parts_name = file_name.split('_')
                for p in parts_name:
                    if p.startswith('sub-'):
                        subject_id = p.replace('sub-', '')
                    elif p.startswith('task-'):
                        task = p.replace('task-', '')

        if subject_id is None:
            subject_id = os.path.basename(file_path).split('_')[0]

        sample_key = f'ADFTD-{subject_id}-{task}_{seg_idx}'
        return sample_key, subject_id, None, task

    @staticmethod
    def coordinate(coord_file_path, channel_names):
        """
        Get 3D coordinates for 10-20 system channels from ELC file
        Modified: 2025-11-16 - Changed to use standard_1005.elc file

        Args:
            coord_file_path: Path to the ELC file (standard_1005.elc)
            channel_names: List of channel names

        Returns:
            np.ndarray: (N, 3) array of XYZ coordinates
        """
        # Use the utils._load_xyz_from_elc function
        return utils._load_xyz_from_elc(coord_file_path, channel_names)

    @staticmethod
    def qc(
        sample: np.ndarray,
        xyz: np.ndarray,
        ch_names: List[str],
        threshold: float = 100.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Quality control disabled for ADFTD dataset
        Modified: 2025-11-16 - Changed to return None (no QC filtering)
        """
        # No QC filtering - return None to skip QC process
        return None

    @staticmethod
    def resample(sample_dict, chs, original_sampling_rate, target_sampling_rate, segment_len):
        """
        For ADFTD, all resampling is already done in shaping function.
        Shaping returns dict with 'data' and 'marker' keys.
        Just extract and return the data.

        Args:
            sample_dict: dict with keys 'data' (19, T_sec, D_hz) and 'marker' (sampling rate)
            chs: number of channels (19)
            original_sampling_rate: original rate (500 Hz)
            target_sampling_rate: target rate (500 Hz, not used since already resampled)
            segment_len: segment length (30 sec, not used)

        Returns:
            np.ndarray: EEG data with shape (19, T_sec, D_hz)
        """
        if isinstance(sample_dict, dict):
            # Extract data from dict returned by shaping
            return sample_dict['data']  # (19, T_sec, D_hz)
        else:
            # Fallback for unexpected format
            return sample_dict

    @staticmethod
    def _load_participants_labels():
        """
        Load participants.tsv and create subject_id -> label mapping
        Returns dict: {'001': 0, '002': 0, ...}

        Label encoding (binary classification):
        - C (Healthy Control) → 0
        - A (Alzheimer's Disease) → 2
        - F (Frontotemporal Dementia) → 1 (EXCLUDED from training)
        """
        import pandas as pd
        import os

        # Try multiple possible paths for participants.tsv
        possible_paths = [
            "/storage/connectome/bohee/DIVER_ADFTD/data/raw/participants.tsv",  # Server path
            r"D:\GitHub\DIVER\data\ADFTD\participants.tsv",
            r"D:\GitHub\DIVER\data\ADFTD\ds004504\participants.tsv",
            "/d/GitHub/DIVER/data/ADFTD/participants.tsv",
            "/d/GitHub/DIVER/data/ADFTD/ds004504/participants.tsv",
        ]

        participants_path = None
        for path in possible_paths:
            if os.path.exists(path):
                participants_path = path
                break

        if participants_path is None:
            print("[Warning] participants.tsv not found. Labels will be None.")
            return {}

        # Read participants.tsv
        df = pd.read_csv(participants_path, sep='\t')

        # Create mapping: subject_id (without 'sub-') -> numeric label
        # Binary classification: HC vs AD only (FTD excluded)
        label_map = {}
        group_to_label = {'C': 0, 'A': 2, 'F': 1}  # Correct mapping from original notebook

        for _, row in df.iterrows():
            participant_id = row['participant_id']  # 'sub-001'
            group = row['Group']  # 'C', 'A', or 'F'

            # Extract numeric ID: 'sub-001' -> '001'
            subject_id = participant_id.replace('sub-', '')
            label = group_to_label.get(group, None)

            # Only include HC and AD (exclude FTD)
            if label is not None and group in ['C', 'A']:
                label_map[subject_id] = label

        print(f"[ADFTD] Loaded {len(label_map)} subject labels (HC + AD only, FTD excluded)")
        print(f"  - HC (0): {sum(1 for v in label_map.values() if v == 0)} subjects")
        print(f"  - AD (2): {sum(1 for v in label_map.values() if v == 2)} subjects")

        return label_map

    @staticmethod
    def label(raw):
        """
        Get label for ADFTD subject from participants.tsv
        ADFTD is a fine-tune dataset with labels: HC, AD, FTD

        Label encoding (binary classification):
        - 0: HC (Healthy Control) - C in participants.tsv
        - 2: AD (Alzheimer's Disease) - A in participants.tsv
        - FTD (F) is EXCLUDED from training

        Args:
            raw: MNE Raw object with file path info

        Returns:
            int: Label (0=HC, 2=AD) or None if not found
        """
        import re

        # Load participants mapping if not cached
        if ADFTDDatasetSetting._label_cache is None:
            ADFTDDatasetSetting._label_cache = ADFTDDatasetSetting._load_participants_labels()

        # Extract subject ID from raw object
        subject_id = None

        # Method 1: Try to get from filenames attribute
        if hasattr(raw, 'filenames') and raw.filenames:
            file_path = str(raw.filenames[0])  # Convert WindowsPath to string
            match = re.search(r'sub-(\d+)', file_path)
            if match:
                subject_id = match.group(1)

        # Method 2: Try from info['description']
        if subject_id is None and 'description' in raw.info:
            description = raw.info['description']
            match = re.search(r'sub-(\d+)', description)
            if match:
                subject_id = match.group(1)

        # Method 3: Try from meas_date or other info
        if subject_id is None and hasattr(raw, '_filenames'):
            for fn in raw._filenames:
                match = re.search(r'sub-(\d+)', fn)
                if match:
                    subject_id = match.group(1)
                    break

        # Get label from cache
        if subject_id and subject_id in ADFTDDatasetSetting._label_cache:
            return ADFTDDatasetSetting._label_cache[subject_id]
        else:
            print(f"[Warning] Could not find label for subject: {subject_id}")
            return None

########################################################################
# End of ADFTD Dataset Setting
########################################################################

class HBNDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'HBN'

    @staticmethod
    def file_list(rootDir, ext_filter, percent):
        file_path_list = []
        walker = tqdm(os.walk(rootDir), desc="Walking through directories", mininterval=10)

        for root, dirs, files in walker:
            for file in files:
                if not file.endswith('.set'):
                    continue
                file_path_list.append(os.path.join(root, file))

        file_path_list = sorted(file_path_list)
        #random.shuffle(file_path_list)
        samples_list_to_process = file_path_list[:int(len(file_path_list) * percent)]
        #samples_list_to_process = file_path_list[:1019] # hardcoding to take 10.000 keys
        return {'all': samples_list_to_process}
    
    @staticmethod
    # default shaping
    def shaping(raw, original_sampling_rate, segment_len):
        array = raw.to_data_frame(scalings={'eeg': 1e6}).values[:, 1:]
        points, chs = array.shape
        if points < segment_len * 10 * original_sampling_rate:
            return None, chs
        trim = points % (segment_len * original_sampling_rate)
        array = array[2*segment_len*original_sampling_rate:-(trim+2*segment_len*original_sampling_rate), :]
        array = array.reshape(-1, segment_len, original_sampling_rate, chs).transpose(0, 3, 1, 2)  # (N,C,T,D)
        return array, chs
    
    @staticmethod
    def filter(raw, highpass, lowpass, notch, notch_auto=True):
        #raw._data /= 1000 # raw.to_data_frame으로 처리했을 때 
        original_sampling_rate = int(raw.info["sfreq"])
        if lowpass is None:
            raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        if notch_auto:
            nyquist = original_sampling_rate / 2 
            harmonics = []
            for base in notch:
                mult = 1 
                while base * mult <= nyquist:
                    harmonics.append(base * mult)
                    mult += 1
            harmonics = sorted(set(harmonics))
            raw.notch_filter(harmonics)
        else:
            raw.notch_filter(notch)
        return raw, original_sampling_rate
    
    @staticmethod
    def set_meta_data(file_path, seg_idx):
        release = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path)))).replace("Release", "")
        subject_id = os.path.basename(os.path.dirname(os.path.dirname(file_path))).replace("sub-", "")
        task_full = os.path.basename(file_path)
        task = task_full.split("task-")[1].replace(".set", "")
        file_name = f"HBN-{release}-{subject_id}-{task}"
        sample_key = f'{file_name}_{seg_idx}'
        return sample_key, subject_id, release, task

    @staticmethod
    def qc(
        sample: np.ndarray,
        xyz: np.ndarray,
        ch_names: List[str], 
        threshold: float = 100.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        채널별 QC → bad channel 삭제 → segment 전체 폐기 or clipping 후 반환
        Inputs:
        sample   : np.ndarray, shape (n_ch, n_sec, fs)
        xyz      : np.ndarray, shape (n_ch, 3)
        ch_names : List[str], length n_ch
        threshold: float
        Returns:
        (cleaned_sample, cleaned_xyz, cleaned_ch_names)
        or (None, None, None) if 전체 segment 폐기
        """
        n_ch, n_sec, fs = sample.shape
        per_ch_size = n_sec * fs        

        # 1) threshold 초과 마스크
        abs_sample = np.abs(sample)
        over_mask = abs_sample > threshold
        num_over = over_mask.sum(axis=(1, 2))             # 채널별 초과 개수

        # 2) bad channel 판정
        bad_ch_mask = num_over >= (per_ch_size *0.0333)       # True = bad
        n_bad = bad_ch_mask.sum()

        # 3) 채널 절반 이상 bad → segment 폐기
        if n_bad >= (n_ch *0.5):
            return None, None, None

        # 4) 남은(“good”) 채널 인덱스
        good_idx = np.nonzero(~bad_ch_mask)[0]

        # 5) cleaned sample 생성 + clipping
        cleaned = sample[good_idx].copy().astype(np.float32)
        cleaned_over = over_mask[good_idx]
        cleaned[cleaned_over] = np.sign(cleaned[cleaned_over]) * threshold

        # 6) xyz, ch_names 필터링
        cleaned_xyz   = xyz[good_idx]
        cleaned_names = [ch_names[i] for i in good_idx]

        return cleaned, cleaned_xyz, cleaned_names

class PEERSDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'PEERS'

    @staticmethod
    def file_list(rootDir, ext_filter, num_seg):
        file_path_list = []
        walker = tqdm(os.walk(rootDir), desc="Walking through directories", mininterval=10)

        for root, dirs, files in walker:
            for file in files:
                if ext_filter:
                    if isinstance(ext_filter, str):
                        if not file.endswith(ext_filter):
                            continue
                    elif isinstance(ext_filter, (list, tuple)):
                        if not any(file.endswith(ext) for ext in ext_filter):
                            continue
                file_path_list.append(os.path.join(root, file))
        file_path_list = sorted(file_path_list)
        #random.shuffle(file_path_list)
        N = len(file_path_list)
        S = 20

        q, r = divmod(N, S)
        k = int(num_seg)

        # Starting index of the k‑th segment
        # The first r segments each have (q + 1) items; the rest have q items each
        start = k * q + min(k, r)
        # size of the k-th segment
        size  = q + (1 if k < r else 0)
        end   = start + size

        # 디버깅용
        # print("for debugging:", N, q, r, type(start), start, type(end), end)

        #samples_list_to_process = file_path_list[start:end]
        samples_list_to_process = file_path_list[:68]
        print(f"Segment {k+1}/{S}: indices {start}…{end-1} (size={size})")
        #samples_list_to_process = file_path_list[:int(len(file_path_list) * percent)]
        return {'all': samples_list_to_process}
    
    @staticmethod
    def qc(
        sample: np.ndarray,
        xyz: np.ndarray,
        ch_names: List[str],
        threshold: float = 100.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        채널별 QC → bad channel 삭제 → segment 전체 폐기 or clipping 후 반환
        Inputs:
        sample   : np.ndarray, shape (n_ch, n_sec, fs)
        xyz      : np.ndarray, shape (n_ch, 3)
        ch_names : List[str], length n_ch
        threshold: float
        Returns:
        (cleaned_sample, cleaned_xyz, cleaned_ch_names)
        or (None, None, None) if 전체 segment 폐기
        """
        n_ch, n_sec, fs = sample.shape
        per_ch_size = n_sec * fs        

        # 1) threshold 초과 마스크
        abs_sample = np.abs(sample)
        over_mask = abs_sample > threshold
        num_over = over_mask.sum(axis=(1, 2))             # 채널별 초과 개수

        # 2) bad channel 판정
        bad_ch_mask = num_over >= (per_ch_size *0.0333)       # True = bad
        n_bad = bad_ch_mask.sum()

        # 3) 채널 절반 이상 bad → segment 폐기
        if n_bad >= (n_ch *0.5):
            return None, None, None

        # 4) 남은(“good”) 채널 인덱스
        good_idx = np.nonzero(~bad_ch_mask)[0]

        # 5) cleaned sample 생성 + clipping
        cleaned = sample[good_idx].copy().astype(np.float32)
        cleaned_over = over_mask[good_idx]
        cleaned[cleaned_over] = np.sign(cleaned[cleaned_over]) * threshold

        # 6) xyz, ch_names 필터링
        cleaned_xyz   = xyz[good_idx]
        cleaned_names = [ch_names[i] for i in good_idx]

        return cleaned, cleaned_xyz, cleaned_names

    @staticmethod
    def filter(raw, highpass, lowpass, notch, notch_auto=True):
        original_sampling_rate = int(raw.info["sfreq"])
        if lowpass is None:
            raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        if notch_auto:
            nyquist = original_sampling_rate / 2 
            harmonics = []
            for base in notch:
                mult = 1 
                while base * mult <= nyquist:
                    harmonics.append(base * mult)
                    mult += 1
            harmonics = sorted(set(harmonics))
            raw.notch_filter(harmonics)
        else:
            raw.notch_filter(notch)
        return raw, original_sampling_rate

    @staticmethod
    def set_meta_data(file_path, seg_idx):
    #def set_meta_data_TUEG(file_path, file_key_list, seg_idx):
        file_name = file_path.split('/')[-1][:-4]
        sample_key = f'PEERS-{file_name}_{seg_idx}'
        #file_key_list.append(sample_key)
        subject_id = file_name.split('_')[0] # set subject_id in PEERS
        return sample_key, subject_id, None, None
    
    @staticmethod
    def channel_list(raw):
        if len(raw.info['ch_names']) == 129:        # 129-channel Geodesic Sensor Net
            channel_list = raw.info['ch_names']
        else:                                       # 128-channel BioSemi headcap
            channel_list = raw.info['ch_names'][0:128]
            assert channel_list == mne.channels.make_standard_montage('biosemi128').ch_names, "Expected channel names do not match Biosemi128 montage."

        return channel_list
    
    @staticmethod
    def coordinate(folder_path: str, picked_subchannels: list) -> np.ndarray:
        if len(picked_subchannels) == 129:        # 129-channel Geodesic Sensor Net
            # montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
            coordinate_xyz = utils._load_xyz_from_sfr(os.path.join(folder_path, 'GSN-HydroCel-129.sfp'))
        else:                                       # 128-channel BioSemi headcap
            # montage = mne.channels.make_standard_montage('biosemi128')
            coordinate_xyz = utils._load_xyz_from_csv_peers(os.path.join(folder_path, 'biosemi128_mni_coords_standardhead.csv'))
        # coordinate_xyz = montage.get_positions()['ch_pos']

        return coordinate_xyz

    @staticmethod
    def loader(file_path):
        '''
        Attempt to load an EDF or BDF file. If loading fails, return None.
        '''
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.edf':
                return mne.io.read_raw_edf(file_path, preload=True)
            elif ext == '.bdf':
                return mne.io.read_raw_bdf(file_path, preload=True)
            else:
                raise ValueError(f"[WARN] Unsupported extension {ext!r} for {file_path}: must be .edf or .bdf")
        except Exception as e:
            # If reading fails, log the error and skip this file
            print(f"[ERROR] Failed to load {file_path}: {e}. Skipping.")
            return None

class HarvardDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'Harvard'

    @staticmethod
    def file_list(rootDir, ext_filter, num_seg):
        file_path_list = []
        walker = tqdm(os.walk(rootDir), desc="Walking through directories", mininterval=10)

        for root, dirs, files in walker:
            for file in files:
                if ext_filter:
                    if isinstance(ext_filter, str):
                        if not file.endswith(ext_filter):
                            continue
                    elif isinstance(ext_filter, (list, tuple)):
                        if not any(file.endswith(ext) for ext in ext_filter):
                            continue
                file_path_list.append(os.path.join(root, file))
        file_path_list = sorted(file_path_list)
        #random.shuffle(file_path_list)
        N = len(file_path_list)
        S = 7  # 7개 묶음으로 나눔

        q, r = divmod(N, S)
        k = int(num_seg)

        # Starting index of the k‑th segment
        # The first r segments each have (q + 1) items; the rest have q items each
        start = k * q + min(k, r)
        # size of the k-th segment
        size  = q + (1 if k < r else 0)
        end   = start + size

        # 디버깅용
        # print("for debugging:", N, q, r, type(start), start, type(end), end)

        samples_list_to_process = file_path_list[start:end]
        print(f"Segment {k+1}/{S}: indices {start}…{end-1} (size={size})")
        #samples_list_to_process = file_path_list[:int(len(file_path_list) * percent)]
        return {'all': samples_list_to_process}
    
    @staticmethod
    def qc(
        sample: np.ndarray,
        xyz: np.ndarray,
        ch_names: List[str],
        threshold: float = 100.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        채널별 QC → bad channel 삭제 → segment 전체 폐기 or clipping 후 반환
        Inputs:
        sample   : np.ndarray, shape (n_ch, n_sec, fs)
        xyz      : np.ndarray, shape (n_ch, 3)
        ch_names : List[str], length n_ch
        threshold: float
        Returns:
        (cleaned_sample, cleaned_xyz, cleaned_ch_names)
        or (None, None, None) if 전체 segment 폐기
        """
        n_ch, n_sec, fs = sample.shape
        per_ch_size = n_sec * fs        

        # 1) threshold 초과 마스크
        abs_sample = np.abs(sample)
        over_mask = abs_sample > threshold
        num_over = over_mask.sum(axis=(1, 2))             # 채널별 초과 개수

        # 2) bad channel 판정
        bad_ch_mask = num_over >= (per_ch_size *0.0333)       # True = bad
        n_bad = bad_ch_mask.sum()

        # 3) 채널 절반 이상 bad → segment 폐기
        if n_bad >= (n_ch *0.5):
            return None, None, None

        # 4) 남은(“good”) 채널 인덱스
        good_idx = np.nonzero(~bad_ch_mask)[0]

        # 5) cleaned sample 생성 + clipping
        cleaned = sample[good_idx].copy().astype(np.float32)
        cleaned_over = over_mask[good_idx]
        cleaned[cleaned_over] = np.sign(cleaned[cleaned_over]) * threshold

        # 6) xyz, ch_names 필터링
        cleaned_xyz   = xyz[good_idx]
        cleaned_names = [ch_names[i] for i in good_idx]

        return cleaned, cleaned_xyz, cleaned_names

    @staticmethod
    def filter(raw, highpass, lowpass, notch, notch_auto=True):
        original_sampling_rate = int(raw.info["sfreq"])
        if lowpass is None:
            raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        if notch_auto:
            nyquist = original_sampling_rate / 2 
            harmonics = []
            for base in notch:
                mult = 1 
                while base * mult <= nyquist:
                    harmonics.append(base * mult)
                    mult += 1
            harmonics = sorted(set(harmonics))
            raw.notch_filter(harmonics)
        else:
            raw.notch_filter(notch)
        return raw, original_sampling_rate

    @staticmethod
    def set_meta_data(file_path, seg_idx):
        file_name = file_path.split('/')[-1][:-4]
        sample_key = f'Harvard-{file_name}_{seg_idx}'
        #file_key_list.append(sample_key)
        subject_id = file_name.split('_')[0] # set subject_id in Harvard-EEG
        return sample_key, subject_id, None, None
    
    @staticmethod
    def channel_list(raw):
        channel_list = raw.info['ch_names']
        return channel_list

    @staticmethod
    def loader(file_path):
        '''
        Attempt to load an EDF file. If loading fails, return None.
        '''
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.edf':
                return mne.io.read_raw_edf(file_path, preload=True)
            else:
                raise ValueError(f"[WARN] Unsupported extension {ext!r} for {file_path}: must be .edf")
        except Exception as e:
            # If reading fails, log the error and skip this file
            print(f"[ERROR] Failed to load {file_path}: {e}. Skipping.")
            return None

class FACEDDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'FACED'

    @staticmethod
    def file_list(root_dir: str, split: dict):
        
        files = sorted([
        os.path.join(root_dir, file)
        for file in os.listdir(root_dir)
        if file.endswith(".pkl")
    ])
        parts = {}
        for key, sl in split.items():
            parts[key] = files[sl]

        return parts

    @staticmethod
    def filter(raw, **kwargs):
        return None
    
    @staticmethod
    def shaping(raw, original_sampling_rate, segment_len):
        return None
    
    @staticmethod
    def qc(sample):
        return None
    
    @staticmethod
    def resample(raw, chs, original_sampling_rate, target_resampling_rate, segment_len):
        eeg = signal.resample(raw, segment_len*target_resampling_rate, axis=1)
        eeg_ = eeg.reshape(32, segment_len, target_resampling_rate)
        return eeg_

    @staticmethod
    def set_meta_data(file_path, seg_idx):
        file = os.path.splitext(file_path)[0]
        subj = os.path.basename(file)
        key = f'{subj}-{seg_idx}'
        return key, subj, None, None

    @staticmethod
    def label(raw):
        return np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])

class PhysioNetDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'PhysioNet'

    @staticmethod
    def file_list(root_dir: str, split: dict):
        tasks = ['04','06','08','10','12','14']
        subj_dirs = sorted([file for file in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, file))])
        all_paths = []
        for subj in subj_dirs:
            for task in tasks:
                p = os.path.join(root_dir, subj, f'{subj}R{task}.edf')
                if os.path.exists(p):
                    all_paths.append(p)
        parts = {}
        for key, sl in split.items():
            parts[key] = all_paths[sl]
        return parts

    @staticmethod
    def filter(raw, highpass, lowpass, notch):
        if getattr(raw.info, 'bads', []):
            raw.interpolate_bads()
        raw.set_eeg_reference(ref_channels='average')
        original_sampling_rate = int(raw.info["sfreq"])
        if lowpass is None:
            raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        raw.notch_filter((notch))
        return raw, int(raw.info['sfreq'])
    
    @staticmethod
    def shaping(raw, orig_sr, segment_len):
        events, event_dict = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_dict,
                            tmin=0,
                            tmax=segment_len - 1./orig_sr,
                            baseline=None,
                            preload=True)
        data_uv = epochs.get_data(units='uV')    # (n_epochs, n_ch, n_times)
        data_uv = data_uv[:, :, -(segment_len*orig_sr):]
        n_epochs, n_ch, _ = data_uv.shape
        data_seg = data_uv.reshape(n_epochs, n_ch, segment_len, orig_sr)
        return data_seg, n_ch
    
    @staticmethod
    def qc(sample):
        return None

    @staticmethod
    def set_meta_data(file_path, seg_idx):
        p = Path(file_path)
        subj = p.parent.name
        task = p.stem.split('_')[-1]
        key = f"{subj}R{task}-{seg_idx}"
        #file_key_list.append(key)
        return key, subj, None, task
    
    @staticmethod
    def label(raw):
        events, _ = mne.events_from_annotations(raw)
        labels = events[:,2]
        return labels

class NSRRnchsdbDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'NSRR_nchsdb'

    @staticmethod
    def file_list(rootDir, ext_filter, percent):
        file_path_list = []

        sleep_data_path = os.path.join(rootDir, "sleep_data")
        if not os.path.isdir(sleep_data_path):
            raise ValueError(f"{sleep_data_path} is not a valid directory")

        # sleep_data 하위의 모든 .edf 파일 수집
        for root, dirs, files in os.walk(sleep_data_path):
            for fname in files:
                if fname.endswith('.edf'):
                    file_path_list.append(os.path.join(root, fname))

        # 정렬, 셔플 및 비율만큼 선택
        file_path_list = sorted(file_path_list)
        #random.shuffle(file_path_list)
        samples_list_to_process = file_path_list[:int(len(file_path_list) * percent)]
        #samples_list_to_process = file_path_list[:8]

        return {'all': samples_list_to_process}

    @staticmethod
    def filter(raw, highpass, lowpass, notch, notch_auto=True):
        original_sampling_rate = int(raw.info["sfreq"])
        if lowpass is None:
            raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        if notch_auto:
            nyquist = original_sampling_rate / 2 
            harmonics = []
            for base in notch:
                mult = 1 
                while base * mult <= nyquist:
                    harmonics.append(base * mult)
                    mult += 1
            harmonics = sorted(set(harmonics))
            raw.notch_filter(harmonics)
        else:
            raw.notch_filter(notch)
        return raw, original_sampling_rate

    @staticmethod
    def set_meta_data(file_path, seg_idx):
    #def set_meta_data_TUEG(file_path, file_key_list, seg_idx):
        file_name = file_path.split('/')[-1][:-4]
        sample_key = f'NSRR-nchsdb-{file_name}_{seg_idx}'
        #file_key_list.append(sample_key)
        subject_id = file_name.split('_')[0] # set subject_id
        return sample_key, subject_id, None, None

    @staticmethod
    def qc(
        sample: np.ndarray,
        xyz: np.ndarray,
        ch_names: List[str], 
        threshold: float = 100.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        채널별 QC → bad channel 삭제 → segment 전체 폐기 or clipping 후 반환
        Inputs:
        sample   : np.ndarray, shape (n_ch, n_sec, fs)
        xyz      : np.ndarray, shape (n_ch, 3)
        ch_names : List[str], length n_ch
        threshold: float
        Returns:
        (cleaned_sample, cleaned_xyz, cleaned_ch_names)
        or (None, None, None) if 전체 segment 폐기
        """
        n_ch, n_sec, fs = sample.shape
        per_ch_size = n_sec * fs        

        # 1) threshold 초과 마스크
        abs_sample = np.abs(sample)
        over_mask = abs_sample > threshold
        num_over = over_mask.sum(axis=(1, 2))             # 채널별 초과 개수

        # 2) bad channel 판정
        bad_ch_mask = num_over >= (per_ch_size *0.0333)       # True = bad
        n_bad = bad_ch_mask.sum()

        # 3) 채널 절반 이상 bad → segment 폐기
        if n_bad >= (n_ch *0.5):
            return None, None, None

        # 4) 남은(“good”) 채널 인덱스
        good_idx = np.nonzero(~bad_ch_mask)[0]

        # 5) cleaned sample 생성 + clipping
        cleaned = sample[good_idx].copy().astype(np.float32)
        cleaned_over = over_mask[good_idx]
        cleaned[cleaned_over] = np.sign(cleaned[cleaned_over]) * threshold

        # 6) xyz, ch_names 필터링
        cleaned_xyz   = xyz[good_idx]
        cleaned_names = [ch_names[i] for i in good_idx]

        return cleaned, cleaned_xyz, cleaned_names

class NSRRmrosDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'NSRR_mros'

    @staticmethod
    def file_list(rootDir, ext_filter, percent):
        file_path_list = []

        mros_data_path = os.path.join(rootDir, "polysomnography/edfs/")
        if not os.path.isdir(mros_data_path):
            raise ValueError(f"{mros_data_path} is not a valid directory")

        # sleep_data 하위의 모든 .edf 파일 수집
        for root, dirs, files in os.walk(mros_data_path):
            for fname in files:
                if fname.endswith('.edf'):
                    file_path_list.append(os.path.join(root, fname))

        # 정렬, 셔플 및 비율만큼 선택
        file_path_list = sorted(file_path_list)
        random.shuffle(file_path_list)
        samples_list_to_process = file_path_list[:int(len(file_path_list) * percent)]

        return {'all': samples_list_to_process}

    @staticmethod
    def filter(raw, highpass, lowpass, notch, notch_auto=True):
        original_sampling_rate = int(raw.info["sfreq"])
        if lowpass is None:
            raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        if notch_auto:
            nyquist = original_sampling_rate / 2 
            harmonics = []
            for base in notch:
                mult = 1 
                while base * mult <= nyquist:
                    harmonics.append(base * mult)
                    mult += 1
            harmonics = sorted(set(harmonics))
            raw.notch_filter(harmonics)
        else:
            raw.notch_filter(notch)
        return raw, original_sampling_rate

    @staticmethod
    def set_meta_data(file_path, seg_idx):
    #def set_meta_data_TUEG(file_path, file_key_list, seg_idx):
        file_name = file_path.split('/')[-1][:-4]
        sample_key = f'NSRR-{file_name}_{seg_idx}'
        #file_key_list.append(sample_key)
        subject_id = file_name.split('-')[-1] # set subject_id
        return sample_key, subject_id, None, None
    
class iEEGFRDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'iEEGFR'

    @staticmethod
    def file_list():
        pass

    @staticmethod
    def filter():
        pass

    @staticmethod
    def qc(sample):
        return super().qc(sample)
    
    @staticmethod
    def shaping(raw, original_sampling_rate, segment_len):
        return super().shaping(raw, original_sampling_rate, segment_len)
    
    @staticmethod
    def set_meta_data():
        pass

class largeiEEGDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'large_iEEG'

    @staticmethod
    def file_list(rootDir, ext_filter, percent):
        file_path_list = []
        pattern = re.compile(r'^Sub\d+$')

        for item in os.listdir(rootDir):
            sub_dir_path = os.path.join(rootDir, item)
            if os.path.isdir(sub_dir_path) and pattern.match(item):
                for fname in os.listdir(sub_dir_path):
                    if fname.endswith('.cnt'):
                        file_path_list.append(os.path.join(sub_dir_path, fname))
        file_path_list = sorted(file_path_list)
        random.shuffle(file_path_list)
        samples_list_to_process = file_path_list[:int(len(file_path_list) * percent)]
        return {'all': samples_list_to_process}

    @staticmethod
    def filter(raw, highpass, lowpass, notch, segment_info, batch_size=50): # segment_info parameter is added
        all_data = []
        current_batch = []

        original_sampling_rate = int(raw.info["sfreq"])
        expected_length = segment_info[0]['end'] - segment_info[0]['start']

        #print('data loading for filtering...')

        # ✅ Step 1: 전체 filter 처리
        filtered_raw = raw.copy().load_data()

        #print('low/highpass filtering...')

        if lowpass is None:
            filtered_raw.filter(l_freq=highpass, h_freq=None, verbose='ERROR')
        else:
            filtered_raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate / 2 - 1), verbose='ERROR')

        #print('notch filtering...')

        filtered_raw.notch_filter(notch, verbose='ERROR')

        # ✅ Step 2: crop per segment
        for segment in tqdm(segment_info, desc="Cropping segments"):
            start = segment['start']
            end = segment['end']
            if end - start != expected_length: # TODO : cut 처리의 경우 모든 segment의 길이가 다르다는 점 고려하기
                continue

            picks = segment['channels']
            data = filtered_raw.get_data(start=start, stop=end, picks=picks, units='uV')
            current_batch.append(data)

            if len(current_batch) >= batch_size:
                all_data.extend(current_batch)
                current_batch = []

        if current_batch:
            all_data.extend(current_batch)

        return all_data, original_sampling_rate
    
    @staticmethod
    def qc(sample):
        return np.max(np.abs(sample)) < 1000 # 1mV
    # TODO : threshold가 바뀔 수 있는 점을 고려하면 좋겠어요
    
    @staticmethod
    def shaping(raw, original_sampling_rate, segment_len):
        reshaped_segments = []

        for segment in raw:
            chs, total_points = segment.shape
            expected_points = segment_len * original_sampling_rate

            if total_points != expected_points:
                continue  # 길이 안 맞으면 skip

            # reshape: (chs, timepoints) → (chs, segment_len, Hz)
            reshaped = segment.reshape(chs, segment_len, original_sampling_rate)
            reshaped_segments.append(reshaped)

        return reshaped_segments, None
    
    @staticmethod
    def resample(raw, chs, original_sampling_rate, target_resampling_rate, segment_len):
        chs, time, orig_hz = raw.shape
        # → reshape to (chs * time, orig_hz)
        flat = raw.reshape(-1, orig_hz)

        # → resample along last axis
        resampled = signal.resample(flat, target_resampling_rate, axis=1)

        # → reshape back to (chs, time, target_sr)
        return resampled.reshape(chs, time, target_resampling_rate)

    @staticmethod
    def set_meta_data(file_path, seg_idx): # now it is just copy of that
    #def set_meta_data_TUEG(file_path, file_key_list, seg_idx):
        file_name = file_path.split('/')[-1][:-4]
        sample_key = f'NSRR_nchsdb-{file_name}_{seg_idx}'
        #file_key_list.append(sample_key)
        subject_id = file_name.split('_')[0] # set subject_id in TUEG
        return sample_key, subject_id, None, None

class FacesBasicDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'FacesBasic'

    @staticmethod
    def file_list(root_dir: str, split: dict):
        assert os.path.exists(root_dir), f'root_path ({root_dir}) does not exist.'

        # point at the data/ directory
        data_dir = os.path.join(root_dir, 'data')
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"'data/' subdirectory not found under {root_dir}")

        # grab every .mat file one level down (i.e. data/<subject>/*.mat)
        pattern = os.path.join(data_dir, '*', '*_faceshouses.mat')
        file_path_list = sorted(glob.glob(pattern))

        parts = {}
        for key, sl in split.items():
            parts[key] = file_path_list[sl]
        print("load file list")
        return parts
    
    @staticmethod
    def channel_list(file_path: str):
        p = Path(file_path)
        subj = p.parent.name
        # p.parents[2] is the “faces_basic” folder, so we go up two levels and then into "locs"
        locs_path = p.parents[2] / "locs" / f"{subj}_xslocs.mat"
        
        # Check if the locs file exists
        if not locs_path.exists():
            raise FileNotFoundError(f"Locs file not found: {locs_path}")
        
        # Load the locs file
        locs_file = scio.loadmat(locs_path)

        # The number of channels is length of 'elcode'
        n = len(locs_file['elcode'])

        # Build list of strings "0", "1", ..., "n-1"
        channel_list = [str(i) for i in range(n)]

        return channel_list

    @staticmethod
    def filter(raw, highpass, lowpass, notch):
        original_sampling_rate = int(raw['srate'][0][0])
        original_data = raw['data']
        #raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))

        filtered_data = utils.filter_ecog_data(original_data, powerline_freq=notch, fs=original_sampling_rate, Qfactor=34.999)
        raw['data'] = filtered_data.astype(float)
        original_sampling_rate = 1000   # for shaping
        
        return raw, original_sampling_rate
    
    @staticmethod
    def shaping(raw, orig_sr=1000, segment_len=1):
        signal = raw['data']    # ECoG signal (timepoints, channels)
        stim = raw['stim']      # stimulus (timepoints, 1)
        chunk_size = orig_sr    # 1000ms = 1s

        # data handling
        label_processor_faces_basic = ContinuousBlockLabelProcessor(stimuli_codes={i for i in range(1,102)}, 
                                    pre_stimuli_code=101, pre_stimuli_duration=400, stimuli_duration=400, truncate_longer_stimuli=False,
                                    ignore_codes={})
        process_result = label_processor_faces_basic.process_label_data(stim)

        # empty lists to hold segmented signals
        signal_list = []

        # iterate through 800 ms segments
        for label, start, end in process_result['epochs']:
            if(end - start != 800):
                print(f"Warning: segment length is not 800ms, but {end - start}ms.")
            seg = signal[start:end]
            # zero-pad to 1000 ms
            P, C = seg.shape
            if P < chunk_size:
                pad_len = chunk_size - P    # 200
                seg = np.vstack([seg, np.zeros((pad_len, C), dtype=signal.dtype)])
            signal_list.append(seg)

        # stack into (N, orig_sr, C)
        X = np.stack(signal_list, axis=0)

        # → (N, C, orig_sr)
        X = X.transpose(0, 2, 1)

        # → (N, C, 1, orig_sr)
        data_seg = X[:, :, np.newaxis, :]
        n_epochs, n_ch, _, _ = data_seg.shape

        return data_seg, n_ch
    
    @staticmethod
    def label(raw):
        stim = raw['stim']      # stimulus (timepoints, 1)

        # data handling
        label_processor_faces_basic = ContinuousBlockLabelProcessor(stimuli_codes={i for i in range(1,102)}, 
                                    pre_stimuli_code=101, pre_stimuli_duration=400, stimuli_duration=400, truncate_longer_stimuli=False,
                                    ignore_codes={})
        process_result = label_processor_faces_basic.process_label_data(stim)
        
        # empty lists to hold segmented labels
        labels = []

        for label, start, end in process_result['epochs']:
            if(end - start != 800):
                print(f"Warning: segment length is not 800ms, but {end - start}ms.")
            labels.append(label)

        # make the stimulus label binary
        for i in range(len(labels)):
            if (labels[i] >=1 and labels[i] <= 50):
                labels[i] = 0
            else:
                labels[i] = 1

        return np.array(labels)

    @staticmethod
    def qc(sample):
        return np.max(np.abs(sample)) < 1000 # 1mV

    @staticmethod
    def set_meta_data(file_path, seg_idx):
        p = Path(file_path)
        subj = p.parent.name
        task = p.stem.split('_')[-1]
        key = f"{subj}_{task}-{seg_idx}"
        return key, subj, None, task
    
class FingerflexDatasetSetting(DefaultDatasetSetting):
    def __init__(self):
        super().__init__()
        self.name = 'Fingerflex'

    @staticmethod
    def file_list(root_dir: str, split: dict):
        assert os.path.exists(root_dir), f'root_path ({root_dir}) does not exist.'

        # point at the data/ directory
        data_dir = os.path.join(root_dir, 'BCI_Competion4_dataset4_data_fingerflexions')
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"'BCI_Competion4_dataset4_data_fingerflexions/' subdirectory not found under {root_dir}")

        # grab every .mat file one level down (i.e. data/<subject>/*.mat)
        comp_pattern = os.path.join(data_dir, '*', '*_comp.mat')
        file_path_list = sorted(glob.glob(comp_pattern))

        testlabels_patterns = os.path.join(data_dir, '*', '*_testlabels.mat')
        test_labels_list = sorted(glob.glob(testlabels_patterns))

        parts = {}
        for key, sl in split.items():
            parts[key] = file_path_list[sl]
        print("load file list")
        return parts

    @staticmethod
    def channel_list(file_path: str):
        p = Path(file_path)
        subj = p.parent.name
        # p.parents[2] is the “faces_basic” folder, so we go up two levels and then into "locs"
        locs_path = p.parents[2] / "locs" / f"{subj}_xslocs.mat"
        
        # Check if the locs file exists
        if not locs_path.exists():
            raise FileNotFoundError(f"Locs file not found: {locs_path}")
        
        # Load the locs file
        locs_file = scio.loadmat(locs_path)

        # The number of channels is length of 'elcode'
        n = len(locs_file['elcode'])

        # Build list of strings "0", "1", ..., "n-1"
        channel_list = [str(i) for i in range(n)]

        return channel_list

    @staticmethod
    def filter(raw, highpass, lowpass, notch):
        original_sampling_rate = int(raw['srate'][0][0])
        #raw.filter(l_freq=highpass, h_freq=min(lowpass, original_sampling_rate/2-1))
        original_data = raw['data']

        filtered_data = utils.filter_ecog_data(original_data, powerline_freq=notch, fs=original_sampling_rate, Qfactor=34.999)
        raw['data'] = filtered_data.astype(float)
        return raw, original_sampling_rate
    
    @staticmethod
    def shaping(raw, orig_sr=1000, segment_len=1):
        signal = raw['data']    # ECoG signal (timepoints, channels)
        stim = raw['stim']      # stimulus (timepoints, 1)
        chunk_size = orig_sr    # 1000ms = 1s

        # data handling
        label_processor_faces_basic = ContinuousBlockLabelProcessor(stimuli_codes={i for i in range(1,102)}, 
                                    pre_stimuli_code=101, pre_stimuli_duration=400, stimuli_duration=400, truncate_longer_stimuli=False,
                                    ignore_codes={})
        process_result = label_processor_faces_basic.process_label_data(stim)

        # empty lists to hold segmented signals
        signal_list = []

        # iterate through 800 ms segments
        for label, start, end in process_result['epochs']:
            if(end - start != 800):
                print(f"Warning: segment length is not 800ms, but {end - start}ms.")
            seg = signal[start:end]
            # zero-pad to 1000 ms
            P, C = seg.shape
            if P < chunk_size:
                pad_len = chunk_size - P    # 200
                seg = np.vstack([seg, np.zeros((pad_len, C), dtype=signal.dtype)])
            signal_list.append(seg)

        # stack into (N, orig_sr, C)
        X = np.stack(signal_list, axis=0)

        # → (N, C, orig_sr)
        X = X.transpose(0, 2, 1)

        # → (N, C, 1, orig_sr)
        data_seg = X[:, :, np.newaxis, :]
        n_epochs, n_ch, _, _ = data_seg.shape

        return data_seg, n_ch

    @staticmethod
    def label(raw):
        flex = raw['flex'].T    # flex shape is (5, timepoints) # TODO : if BCIC IV, change struct name

        seg_point = 0
        seg_len = 1000

        whole_len = len(flex.T)
        labels = []

        while seg_point + seg_len < whole_len:
            current_flex = flex[:,seg_point:seg_point+seg_len]
            #print(flex[:,seg_point:seg_point+seg_len])
            labels.append(current_flex)
            
            seg_point += seg_len

        return np.array(labels)

    @staticmethod
    def qc(sample):
        return None

    @staticmethod
    def set_meta_data(file_path, seg_idx):
        p = Path(file_path)
        subj = p.parent.name
        task = p.stem.split('_')[-1]
        key = f"{subj}_{task}-{seg_idx}"
        return key, subj, None, task


####### registry to store all dataset-specific information #######
_REGISTRY = {
        'TUEG': dict(ext=".edf",
            modality='EEG',
            loader=lambda x: mne.io.read_raw_edf(x, preload=True),
            channel_list = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'],
            channel_sets_in_raw={
                '01_tcp_ar': [
                        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                        'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
                        'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
                ],
                '02_tcp_le': [
                        'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                        'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                        'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'
                ],
                '03_tcp_ar_a': [
                        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                        'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
                        'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
                ]
            },
            coordinate_function=utils._load_xyz_from_elc,
            train_type="pretrain",
            split = None,
            file_list_function = TUEGDatasetSetting.file_list,
            filter_function=TUEGDatasetSetting.filter,
            shape_function=TUEGDatasetSetting.shaping,
            qc_process=TUEGDatasetSetting.qc,
            resample_function=TUEGDatasetSetting.resample,
            set_meta_data_function=TUEGDatasetSetting.set_meta_data,
            label_function=TUEGDatasetSetting.label),

        'HBN': dict(ext=".set",
            modality='EEG',
            loader=lambda x: mne.io.read_raw_eeglab(x, preload=True),
            channel_list=[f"E{i}" for i in range(1, 129)] + ["Cz"],
            coordinate_function=utils._load_xyz_from_sfr,
            train_type="pretrain",
            split = None,
            file_list_function = HBNDatasetSetting.file_list,
            filter_function=HBNDatasetSetting.filter,
            shape_function=HBNDatasetSetting.shaping,
            qc_process=HBNDatasetSetting.qc,
            resample_function=HBNDatasetSetting.resample,
            set_meta_data_function=HBNDatasetSetting.set_meta_data,
            label_function=HBNDatasetSetting.label),

        ########################################################################
        # ADFTD Registry Entry - Added 2025-10-17
        # OpenNeuro dataset ds004504: AD/FTD EEG recordings
        ########################################################################
        'ADFTD': dict(ext=".set",
            modality='EEG',
            loader=lambda x: mne.io.read_raw_eeglab(x, preload=True),
            channel_list=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                          'T7', 'C3', 'Cz', 'C4', 'T8',
                          'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'],  # 10-20 system (19 channels)
            coordinate_function=utils._load_xyz_from_elc,  # Use standard_1005.elc file
            train_type="finetune",
            # LEAD paper: Stratified random split with seed=42
            # Binary classification: HC vs AD only (FTD excluded)
            # HC (29 subjects): ~17 train, ~6 val, ~6 test
            # AD (36 subjects): ~21 train, ~7 val, ~8 test
            # Split done in file_list function, this dict is passed but ignored
            split = {
                'train': slice(0, 53),   # Placeholder (actual split in file_list)
                'val': slice(53, 71),    # Placeholder
                'test': slice(71, 88)    # Placeholder
            },
            file_list_function = ADFTDDatasetSetting.file_list,
            filter_function=ADFTDDatasetSetting.filter,
            shape_function=ADFTDDatasetSetting.shaping,
            qc_process=None,
            resample_function=ADFTDDatasetSetting.resample,
            set_meta_data_function=ADFTDDatasetSetting.set_meta_data,
            label_function=ADFTDDatasetSetting.label),

        'PEERS': dict(ext=[".edf", ".bdf"],
            modality='EEG',
            loader=lambda x: PEERSDatasetSetting.loader(x),
            channel_list=PEERSDatasetSetting.channel_list,
            coordinate_function=PEERSDatasetSetting.coordinate,
            train_type="pretrain",
            split = None,
            file_list_function = PEERSDatasetSetting.file_list,
            filter_function=PEERSDatasetSetting.filter,
            shape_function=PEERSDatasetSetting.shaping,
            qc_process=PEERSDatasetSetting.qc,
            resample_function=PEERSDatasetSetting.resample,
            set_meta_data_function=PEERSDatasetSetting.set_meta_data,
            label_function=PEERSDatasetSetting.label),

        'Harvard': dict(ext=".edf",
            modality='EEG',
            loader=lambda x: HarvardDatasetSetting.loader(x),
            channel_list=HarvardDatasetSetting.channel_list,
            raw_channel_list=mne.channels.make_standard_montage('standard_1020').ch_names,
            coordinate_function=utils._load_xyz_from_elc,
            train_type="pretrain",
            split = None,
            file_list_function = HarvardDatasetSetting.file_list,
            filter_function=HarvardDatasetSetting.filter,
            shape_function=HarvardDatasetSetting.shaping,
            qc_process=HarvardDatasetSetting.qc,
            resample_function=HarvardDatasetSetting.resample,
            set_meta_data_function=HarvardDatasetSetting.set_meta_data,
            label_function=HarvardDatasetSetting.label),
        
        'FACED': dict(ext=".pkl",
            modality='EEG',
            loader=lambda x: pickle.load(open(x, "rb")),
            channel_list=['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'Fc1', 'Fc2', 'Fc5', 
                            'Fc6', 'Cz', 'C3', 'C4', 'T7', 'T8', 'A1', 'A2', 'Cp1', 'Cp2', 
                            'Cp5', 'Cp6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'Po3', 'Po4', 'Oz', 
                            'O1', 'O2'],
            coordinate_function=utils._load_xyz_from_elc,
            train_type="finetune",
            split=dict(train=slice(0,80),
                       val=slice(80,100),
                       test=slice(100,None)),
            file_list_function = FACEDDatasetSetting.file_list,
            filter_function = None,
            shape_function   = None,
            qc_process      = None,
            resample_function = FACEDDatasetSetting.resample,
            set_meta_data_function = FACEDDatasetSetting.set_meta_data,
            label_function        = FACEDDatasetSetting.label,),

        'PhysioNet': dict(ext=".edf",
            modality='EEG',
            loader=lambda x: mne.io.read_raw_edf(x, preload=True),
            channel_list=['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 
                            'Cz', 'C2', 'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 
                            'Cp6', 'Fp1', 'Fpz', 'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 
                            'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'Ft7', 'Ft8', 
                            'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1', 
                            'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
                            'O1', 'Oz', 'O2', 'Iz'],
            raw_channel_list=['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                     'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                     'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                     'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                     'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                     'O1..', 'Oz..', 'O2..', 'Iz..'],
            coordinate_function=utils._load_xyz_from_elc,
            train_type="finetune",
            split=dict(train=slice(0,70),
                       val=slice(70,89),
                       test=slice(89,109)),
            file_list_function = PhysioNetDatasetSetting.file_list,
            filter_function = PhysioNetDatasetSetting.filter,
            shape_function  = PhysioNetDatasetSetting.shaping,
            qc_process      = None,
            resample_function= PhysioNetDatasetSetting.resample,
            set_meta_data_function = PhysioNetDatasetSetting.set_meta_data,
            label_function        = PhysioNetDatasetSetting.label,),

        'NSRR_nchsdb': dict(ext='*.edf',
            modality='EEG',
            loader=lambda x: mne.io.read_raw_edf(x, preload=True),
            channel_list=['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'Cz'],
            raw_channel_list=['EEG F3-M2', 'EEG F4-M1', 'EEG C3-M2', 'EEG C4-M1', 'EEG O1-M2', 'EEG O2-M1', 'EEG CZ-O1',
                              'EEG O1', 'EEG O2', 'EEG C3', 'EEG C4', 'EEG F3', 'EEG F4', 'EEG CZ',
                                'O1', 'O2', 'C3', 'C4', 'F3', 'F4', 'CZ'],
            coordinate_function=utils._load_xyz_from_elc,
            train_type="pretrain",
            split=None,
            file_list_function = NSRRnchsdbDatasetSetting.file_list,
            filter_function = NSRRnchsdbDatasetSetting.filter,
            shape_function  = NSRRnchsdbDatasetSetting.shaping,
            qc_process      = NSRRnchsdbDatasetSetting.qc,
            resample_function= NSRRnchsdbDatasetSetting.resample,
            set_meta_data_function = NSRRnchsdbDatasetSetting.set_meta_data,
            label_function        = None,),

        'NSRR_mros': dict(ext='*.edf',
            modality='EEG',
            loader=lambda x: mne.io.read_raw_edf(x, preload=True),
            channel_list=['C3', 'C4'],
            coordinate_function=utils._load_xyz_from_elc,
            train_type="pretrain",
            split=None,
            file_list_function = NSRRmrosDatasetSetting.file_list,
            filter_function = NSRRmrosDatasetSetting.filter,
            shape_function  = NSRRmrosDatasetSetting.shaping,
            qc_process      = NSRRmrosDatasetSetting.qc,
            resample_function= NSRRmrosDatasetSetting.resample,
            set_meta_data_function = NSRRmrosDatasetSetting.set_meta_data,
            label_function        = None,),
            
        'large_iEEG': dict(ext="*.cnt",
            modality='iEEG',
            loader=lambda x: mne.io.read_raw_cnt(x, preload=False, verbose=False),
            channel_list=None, # it should be changed
            coordinate_function=utils._load_xyz_from_csv,
            train_type="pretrain",
            split=None,
            file_list_function = largeiEEGDatasetSetting.file_list,
            filter_function = largeiEEGDatasetSetting.filter,
            shape_function  = largeiEEGDatasetSetting.shaping,
            qc_process      = None,
            resample_function= largeiEEGDatasetSetting.resample,
            set_meta_data_function = largeiEEGDatasetSetting.set_meta_data,
            label_function        = None,
            set_channel_meta_data_function = utils._load_xyz_meta_data_from_csv,
            ),
            
        'iEEG_FR': dict(ext="*.edf",
            modality='iEEG',
            loader=lambda x: mne.io.read_raw_edf(x, preload=False, verbose=False),
            channel_list=None, # it should be changed
            coordinate_function=utils._load_xyz_from_csv,
            train_type="pretrain",
            split=None,
            file_list_function = iEEGFRDatasetSetting.file_list,
            filter_function = iEEGFRDatasetSetting.filter,
            shape_function  = iEEGFRDatasetSetting.shaping,
            qc_process      = iEEGFRDatasetSetting.qc,
            resample_function= iEEGFRDatasetSetting.resample,
            set_meta_data_function = iEEGFRDatasetSetting.set_meta_data,
            label_function        = None,
            set_channel_meta_data_function = utils._load_xyz_meta_data_from_csv,
            ),
        
        'FacesBasic': dict(ext=".mat",
            modality='iEEG',
            loader=lambda x: scio.loadmat(x),
            channel_list=FacesBasicDatasetSetting.channel_list,
            coordinate_function=utils._load_xyz_from_mat,
            train_type="finetune",
            split=dict(train=slice(0,10),
                       val=slice(10,12),
                       test=slice(12,None)),
            file_list_function = FacesBasicDatasetSetting.file_list,
            filter_function = FacesBasicDatasetSetting.filter,
            shape_function   = FacesBasicDatasetSetting.shaping,
            qc_process      = FacesBasicDatasetSetting.qc,
            resample_function = FacesBasicDatasetSetting.resample,
            set_meta_data_function = FacesBasicDatasetSetting.set_meta_data,
            label_function        = FacesBasicDatasetSetting.label,)}


####### REGISTRY reader for main code #######
class DatasetRegistryReader():
    def __init__(self, dataset_name, data_path):
        if dataset_name not in _REGISTRY:
            raise ValueError(f"Unknown dataset {dataset_name}")
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.datasetcfg = _REGISTRY[dataset_name]
    
    def _return_read_function(self, data_path):
        "set the proper read function for each dataset"
        return self.datasetcfg["loader"](data_path)

    def _set_channel_list(self, data_path=None) -> list:
        return self.datasetcfg['channel_list']
        
    def _coordinate(self, coord_file_path, picked_subchannels):
        func = self.datasetcfg["coordinate_function"]
        if picked_subchannels:
            want = picked_subchannels
            if self.dataset_name == 'NSRR_nchsdb':
                want = []
                for picked_subchannel in picked_subchannels:
                    parts = picked_subchannel.split()
                    if len(parts) == 2:
                        pure_channel_name = picked_subchannel.split('-')[0].split(' ')[1]
                    else:
                        pure_channel_name = picked_subchannel
                    want.append(pure_channel_name)
            if self.dataset_name == 'PEERS':
                return func(coord_file_path, picked_subchannels)
        else:
            want = self._set_channel_list()
        return func(coord_file_path, want)

    def _train_type(self):
        return self.datasetcfg["train_type"]

    def _file_ext(self):
        return self.datasetcfg["ext"]
    
    def _modality(self):
        return self.datasetcfg["modality"]

    def _filter_function(self):
        return self.datasetcfg.get("filter_function")

    def _shape_function(self):
        return self.datasetcfg.get("shape_function")

    def _qc_process(self):
        return self.datasetcfg.get("qc_process")

    def _resample_function(self):
        return self.datasetcfg.get("resample_function")
    
    def _set_meta_data(self):
        return self.datasetcfg.get("set_meta_data_function")
    
    def _get_file_list(self):
        return self.datasetcfg.get("file_list_function")

    def _get_split_dict(self):
        return self.datasetcfg['split']

    def _get_labels(self):
        return self.datasetcfg.get("label_function")

    def _get_channel_meta_data(self):
        return self.datasetcfg.get("set_channel_meta_data_function")

    def _delete_bad_channels(self):
        return self.datasetcfg.get("bad_channel_delete")

    def _unit_change(self):
        return self.datasetcfg.get('unit_change')
    
    def _channel_pick(self, raw, path):
        picked_channels = None
        if "channel_sets_in_raw" in self.datasetcfg:
            for key, chs in self.datasetcfg["channel_sets_in_raw"].items():
                if key in path:
                    if all(ch in raw.info["ch_names"] for ch in chs):
                        raw.pick_channels(chs, ordered=True)
                        return raw, picked_channels
                    return None, None
            return None, None
        if "raw_channel_list" in self.datasetcfg:
            chs = self.datasetcfg["raw_channel_list"]
            existing_chs = [ch for ch in chs if ch in raw.info["ch_names"]]
            picked_channels = existing_chs
            if existing_chs:
                raw.pick_channels(existing_chs, ordered=True)
            return raw, picked_channels
        elif self.dataset_name == 'PEERS':
            channel_fn = self._set_channel_list()
            picked_channels = channel_fn(raw)
            if len(raw.info['ch_names']) != 129:
                raw.pick_channels(picked_channels, ordered=True)
            return raw, picked_channels
        else:
            return raw, picked_channels
