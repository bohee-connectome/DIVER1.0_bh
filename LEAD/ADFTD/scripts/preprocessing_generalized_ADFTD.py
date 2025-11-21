import os
import random

import mne
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import scipy
from scipy import signal
import pickle
import lmdb
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from joblib import Parallel, delayed
import shutil
from contextlib import nullcontext
import time

import pyprep

########################################################################
# ADFTD Modification - 2025-10-17
# Changed to use ADFTD-specific dataset setting file
########################################################################
import preprocessing_generalized_datasetsetting_ADFTD as DatasetSetting

import argparse

def none_or_float(value):
    if value.lower() == 'none':
        return None
    return float(value)

parser = argparse.ArgumentParser(description='Preprocess TUEG data differently')
parser.add_argument('--parallel', action='store_true', help='use parallel processing')
parser.add_argument('--n_jobs', type=int, default=None, help='Number of parallel jobs if parallel processing is enabled')
parser.add_argument('--resample_rate', type=int, default=500, help='set resample rate')
parser.add_argument('--highpass', type=float, default=0.3, help='set high pass filter')
parser.add_argument('--lowpass', type=none_or_float, default=200, help='set low pass filter')
parser.add_argument('--notch_filter', nargs='+', type=float, default=[60], help='set notch filter(should not be changed)')
parser.add_argument('--percent', type=float, default=0.1, help='set the percent of data')
parser.add_argument('--segment_len', type=int, default=30, help='time segment length of each eeg array')
parser.add_argument('--dataset_name', default='TUEG', help = 'set the name of dataset to preprocess')
parser.add_argument('--data_path', default=None, help='set the raw data directory')
parser.add_argument('--save_path_parent', default=None, help='set the save path parent')
parser.add_argument('--coordinate_file_path', default=None, help='set coordinate file path. it should be sfr or elc')
parser.add_argument('--num_chunks', type=int, default=10, help='set chunks of file path')
parser.add_argument('--shape_version', type=str, default='v1', choices=['v1', 'v2'],
                    help='shaping version: v1=(19,30,500), v2=multi-scale (19,1,500)/(19,2,250)/(19,4,125)')

params = parser.parse_args()

class preprocessing_data():
    def __init__(self, dataset = None, percent = params.percent, highpass = params.highpass, lowpass = params.lowpass,
        notch_filter = params.notch_filter, resample_rate = params.resample_rate, segment_len = params.segment_len,
        data_path = None, save_path_parent = None, version = params.shape_version, resume = False, completion_path=None):
        """
        main code for overall preprocessing 

        dataset : the name of dataset (e.g., 'TUEG', 'HBN')
        percent : the percent of dataset to use
        highpass : highpass frequency for preprocessing
        lowpass : lowpass frequency for preprocessing
        notch_filter : notch filter harmonic for preprocessing
        resample_rate : resampling rate
        segment_len : time segment length of eeg array
        version : shaping version ('v1' or 'v2')
        data_path : the path that raw data is 
        save_path_parent : the parent path that the saving lmdb file
        resume : use resume(Not yet implemented)
        completion_path : the path that the completion check file save"""

        if resume : 
            raise NotImplementedError("resume not yet implemented")
            
        #self.dataset_class = DatasetRegistryReader(dataset, data_path)
        self.dataset_class = DatasetSetting.DatasetRegistryReader(dataset, data_path)
        self.dataset = dataset
        self.percent = percent
        self.highpass = highpass
        self.lowpass = lowpass
        self.notch_filter = notch_filter
        self.resample_rate = resample_rate
        self.segment_len = segment_len
        self.version = version
        self.data_path = data_path    
        self.save_path_parent = save_path_parent
        self.resume = resume
        self.completion_file = completion_path

        # if use chunk system, it should be implemented
        #num_samples = self._get_all_samples

    def _get_sample_list_to_process(self, data_path, ext_filter, percent, split):
        print("get_sample_list start")
        if self.dataset_class._train_type() == 'pretrain':
            file_list_fn = self.dataset_class._get_file_list()
            return file_list_fn(data_path, ext_filter, percent)
        if self.dataset_class._train_type() == 'finetune':
            file_list_fn = self.dataset_class._get_file_list()
            return file_list_fn(data_path, split)

    def _file_list_chunk(self, data_subpath_list, num_chunks):
        q, r = divmod(len(data_subpath_list), num_chunks)
        boundaries = []
        start = 0
        for i in range(num_chunks):
            end = start + q + (1 if i < r else 0)
            boundaries.append(data_subpath_list[start:end])
            start = end
        chunks = boundaries
        for i, ch in enumerate(chunks):
            s = sum(os.path.getsize(p) for p in ch)
            print(f"Chunk {i}: {len(ch)} files, {s/1e9:.2f} GB")
        return chunks

    def _file_capacity_chunk(self, data_subpath_list, num_chunks, target=6, max_allowed=10):
        # target: limit the capacity of single chunk as 6 GB to prevent OOM
        # max_allowed: if the chunk size is larger than this value, exclude the chunk

        files = [(p, os.path.getsize(p)) for p in data_subpath_list]

        chunks = []
        cur_chunk = []
        cur_sum = 0
        for path, size in files:
            # Adjust threshold crossing based on the number of chunks and remaining files
            if cur_sum + size > target * (1024 ** 3) and len(chunks) < num_chunks - 1:
                chunks.append(cur_chunk)
                cur_chunk = [path]
                cur_sum = size
            else:
                cur_chunk.append(path)
                cur_sum += size

        chunks.append(cur_chunk)
        
        # Validate chunk sizes and filter out oversized chunks
        valid_chunks = []
        for i, ch in enumerate(chunks):
            s = sum(os.path.getsize(p) for p in ch)
            size_gb = s / 1e9
            if size_gb > max_allowed:
                # Save excluded chunk file paths and sizes to a .txt file
                excluded_filename = "excluded_chunk.txt"
                with open(excluded_filename, "a") as f:
                    for p in ch:
                        f.write(f"{p} ({os.path.getsize(p)/1e9:.2f} GB)\n")
                print(f"[Excluded] Chunk {i}: {len(ch)} files, {size_gb:.2f} GB (details saved to {excluded_filename})")
            else:
                print(f"Chunk {i}: {len(ch)} files, {size_gb:.2f} GB")
                valid_chunks.append(ch)

        return valid_chunks
    
    def chunk_dict(self, file_dict, num_chunks):
        if self.dataset == 'Harvard' or self.dataset == 'PEERS':
            chunk_dict = {
                subset: self._file_capacity_chunk(
                    file_list,
                    num_chunks=num_chunks,
                )
                for subset, file_list in file_dict.items()
            }
            total_files = len(file_dict['all'])
            total_elements = sum(len(sub) for sub in chunk_dict['all'])
            total_bytes = sum(os.path.getsize(p) for p in file_dict['all'])
            total_gb = total_bytes / (1024 ** 3)
            print(f"number of chunks: {len(chunk_dict['all'])}")
            print(f"total files: {total_files}")
            print(f"total chunk elements: {total_elements}")
            print(f"total size: {total_gb:.2f} GB")
        else:
            chunk_dict = {
                subset: self._file_list_chunk(
                    file_list,
                    num_chunks=num_chunks,
                )
                for subset, file_list in file_dict.items()
            }
        return chunk_dict
    
    def _preprocess(self, file_path, subset, target_sampling_rate):
    #def _preprocess(self, file_path, file_key_list, db, subset, target_sampling_rate):
        """
        main preprocess code
        file_path : each raw file(e.g., edf, set) path to preprocess
        file_key_list : the list that will be lmdb key
        db : the db path to save the file
        target_sampling_rate : resampling rate
        """

        filter_fn = self.dataset_class._filter_function()
        shape_fn = self.dataset_class._shape_function()

        print("before read is ok")
        print(file_path)
        raw = self._read(file_path)
        if raw is None:
            print(f"[Warning] Failed to read file: {file_path}")
            return []

        raw, picked_subchannels = self.dataset_class._channel_pick(raw, file_path)

        if raw is None:
            print(f"[Warning] Channel pick failed: {file_path}")
            return []
        
        #raw_for_speed = None
        
        #print(raw)
        #print(raw.shape)
        
        if filter_fn is not None:
            raw, original_sampling_rate = filter_fn(raw, self.highpass, self.lowpass, self.notch_filter)

        # TODO : 지금 구조대로면 filter에서 unit 변환 이런거 해야 하는데 그 전에 성진쌤 데이터 get_data하면서 잘 잘라오고 싶어요
        # 지금대로 해도 문제는 없지만 나중에 문제 생길 거 같아요 이 구조 어떻게 더 이쁘게 짤지 고민했으면 좋겠어요

        #print(raw.shape)

        if shape_fn is not None:
            # Pass version parameter for v1/v2 selection
            eeg_array, chs = shape_fn(raw, original_sampling_rate, self.segment_len, self.version)
        else:
            eeg_array = raw
            _, chs, timedim = eeg_array.shape
            original_sampling_rate = timedim // self.segment_len

        #print(eeg_array.shape)

        if eeg_array is None:
            return []
        
        # original
        '''xyz_array = self.dataset_class._coordinate(params.coordinate_file_path)'''

        #sample_key, value_to_store = self._make_value_to_store(file_path, raw, eeg_array, file_key_list, chs, original_sampling_rate, target_sampling_rate,
        #                                    xyz_array, db, subset)

        # original
        '''result = self._make_value_to_store(file_path, raw, eeg_array, chs, original_sampling_rate, 
                                                               target_sampling_rate, xyz_array, subset)'''
        result = self._make_value_to_store(file_path, raw, eeg_array, chs, original_sampling_rate,
                                                               target_sampling_rate, picked_subchannels, subset)
        if result is None:
            return []

        #sample_key, value_to_store = result

        #return subset, sample_key, value_to_store
        return [(subset, sample_key, value_to_store) for sample_key, value_to_store in result]

    def _read(self, data_path):
        raw = self.dataset_class._return_read_function(data_path)
        return raw
    
    # original
    '''def _make_value_to_store(self, file_path, raw, eeg_array, chs, original_sampling_rate,
                            target_sampling_rate, xyz_array, subset):'''
    def _make_value_to_store(self, file_path, raw, eeg_array, chs, original_sampling_rate,
                            target_sampling_rate, picked_subchannels, subset):    
    #def _make_value_to_store(self, file_path, raw, eeg_array, file_key_list, chs, original_sampling_rate,
    #                        target_sampling_rate, xyz_array, db, subset):
        """
        make value to store dictionary and save it to db
        """
        # set labels
        label_function = self.dataset_class._get_labels()
        if label_function is not None:
            label_value = label_function(raw)  # May return single int (subject-level) or None
            if label_value is None:
                labels = [None] * len(eeg_array)
            elif isinstance(label_value, (int, float, np.integer, np.floating)):
                # Subject-level label: assign same label to all segments
                labels = [label_value] * len(eeg_array)
            elif isinstance(label_value, (list, np.ndarray)):
                # Segment-level labels (legacy support)
                labels = label_value
            else:
                labels = [None] * len(eeg_array)
        else:
            labels = [None] * len(eeg_array)

        results = []
        initial_count = len(eeg_array)  # the number of samples before qc
        print(f"Initial sample count: {initial_count}")
        qc_filtered = 0                 # QC에서 걸러진 샘플 수 카운터 (for iEEG)

        for i, (sample, label) in enumerate(zip(eeg_array, labels)):
            # 0) set xyz_array in for loop
            #coordinate_path = params.coordinate_file_path
            coordinate_path = params.coordinate_file_path if hasattr(params, 'coordinate_file_path') and params.coordinate_file_path else file_path
            xyz_array = self.dataset_class._coordinate(coordinate_path, picked_subchannels)
            
            # 1) resampling
            resample_fn = self.dataset_class._resample_function()

            if resample_fn is None:
                sample_rs = sample
            sample_rs = resample_fn(sample, chs, original_sampling_rate, 
                                                                target_sampling_rate, self.segment_len)
                
            # 2) set meta data
            meta_data_fn = self.dataset_class._set_meta_data()
            #sample_key, subject_id, release, task = meta_data_fn(file_path, file_key_list, i)
            sample_key, subject_id, release, task = meta_data_fn(file_path, i)
            
            # get channel list
            channel_fn = self.dataset_class._set_channel_list()
            if self.dataset == 'NSRR_nchsdb' or self.dataset == 'Harvard':
                channel_names = picked_subchannels 
            elif self.dataset_class._modality() == 'EEG':
                if isinstance(channel_fn, (list, tuple)):
                    # it’s already the list of names
                    channel_names = channel_fn
                elif callable(channel_fn):
                    # it’s a function
                    channel_names = channel_fn(raw)
            elif self.dataset_class._modality() == 'iEEG':
                # if iEEG, need file path
                channel_names = channel_fn(file_path)
            #txn = db.begin(write=True)

            #tmin = i*self.segment_len
            #tmax = tmin + self.segment_len

            #raw_seg_qc = raw_for_speed.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False).load_data() # original pyprep code
            #raw_seg_qc = None

            #import pdb;pdb.set_trace()
            #exit()

            # 1)  QC process # set to clip, not default
            qc = self.dataset_class._qc_process()
            if qc is not None:
                result, xyz_array, channel_names = qc(sample_rs, xyz_array, channel_names)
                if result is None:
                    continue
                else:
                    sample_rs = result
            # CBraMod qc
            '''if qc is not None:
                result = qc(sample_rs, xyz_array, channel_names)
                if result == False:
                    continue'''

            # 4) set channel meta data (iEEG case)
            if self.dataset_class._modality() == 'iEEG':
                channel_meta_data_fn = self.dataset_class._get_channel_meta_data()
                channel_names, type_list, group_list, depthlabel_list, bad_list = channel_meta_data_fn(file_path) 

                # 4) make value to store
                value_to_store = {
                    "sample": sample_rs,
                    "label": label,
                    "data_info": { 
                        "Dataset": self.dataset,
                        "modality": self.dataset_class._modality(),
                        "release": release,            
                        "subject_id": subject_id,          
                        "task": task,
                        "resampling_rate": params.resample_rate,
                        "original_sampling_rate": original_sampling_rate,                 
                        "segment_index": i,                
                        "start_time": i * self.segment_len,
                        "channel_names": channel_names,
                        "xyz_id": xyz_array,
                        'channel_type': type_list,
                        'channel_group': group_list,
                        'channel_depthlabel': depthlabel_list,
                        'bad_channel': bad_list,
                    }
                }

            else:   # EEG case
                value_to_store = {
                    "sample": sample_rs,
                    "label": label,
                    "split": subset,
                    "data_info": {
                        "Dataset": self.dataset,
                        "modality": self.dataset_class._modality(),
                        "release": release,
                        "subject_id": subject_id,
                        "task": task,
                        "resampling_rate": params.resample_rate,
                        "original_sampling_rate": original_sampling_rate,
                        "segment_index": i,
                        "start_time": i * self.segment_len,
                        "channel_names": channel_names,
                        "xyz_id": xyz_array,
                    }
                }

            #txn.put(key=sample_key.encode(), value=pickle.dumps(value_to_store))
            #txn.commit()
            #return sample_key, value_to_store
            results.append((sample_key, value_to_store))
        print(f"Total samples: {initial_count}, QC-filtered samples: {qc_filtered}, Passed samples: {len(results)}")    # for iEEG

        return results
    
    def run(self, dataset, subset, data_path) :

        lmdb_save_path = os.path.join(self.save_path_parent, f'{self.percent}_{dataset}', f'{subset}_resample-{self.resample_rate}_highpass-{self.highpass}_lowpass-{self.lowpass}.lmdb')
        os.makedirs(Path(lmdb_save_path).parent, exist_ok=True)
        # Reduced map_size for testing (1GB instead of 16TB)
        db = lmdb.open(lmdb_save_path, map_size=10737418240)  # 10GB

        file_key_list = [] if self.dataset_class._train_type() == "pretrain" else {'train': [], 'val': [], 'test': []}
        #file_key_list = []

        pbar = tqdm(total=len(data_path), desc=f"Processing {dataset}_{subset}")

        if params.parallel:
            with tqdm_joblib(pbar):
                    results = Parallel(n_jobs=params.n_jobs)(
                        delayed(self._preprocess)(file_path, subset, self.resample_rate)
                        for file_path in data_path
                    )
        else:
            results = []
            for file_path in data_path:
                results.append(self._preprocess(file_path, subset, self.resample_rate))
                pbar.update(1)

        pbar.close()

        # below code is past running with incorrect implementation of pbar
        """
        parallel_ctx = tqdm(total=len(data_path), desc=f"Processing {dataset}_{subset}") if params.parallel else nullcontext()
        with parallel_ctx as pbar:
            if params.parallel:
                with tqdm_joblib(pbar):
                    results = Parallel(n_jobs=params.n_jobs)(
                        delayed(self._preprocess)(file_path, subset, self.resample_rate)
                        for file_path in data_path
                    )
            else:
                results = []
                for file_path in data_path:
                    results.append(self._preprocess(file_path, subset, self.resample_rate))
                    #pbar.update(1)
        """

        results = [r for r in results if r]

        flat_results = []
        for r in results:
            if isinstance(r, list):
                flat_results.extend(r)
            else:
                flat_results.append(r)

        # LMDB 저장
        with db.begin(write=True) as txn:
            for sb, key, val in flat_results:
                txn.put(key=key.encode(), value=pickle.dumps(val))
                if isinstance(file_key_list, dict):
                    file_key_list[sb].append(key)
                else:
                    file_key_list.append(key)

            txn.put(b'__keys__', pickle.dumps(file_key_list))

        db.close()
        # self._mark_completion(dataset, subset) # wrong position


        '''for file_path in tqdm(data_path, desc=f"Processing {dataset}_{subset}"):
                self._preprocess(file_path, file_key_list, db, subset, target_sampling_rate=self.resample_rate)
            
        with db.begin(write=True) as txn:
            txn.put(key=b'__keys__', value=pickle.dumps(file_key_list))

        db.close()

        self._mark_completion(dataset)'''

    def _mark_completion(self, dataset, subset, data_path):
        """
        Create a file to mark training completion.
        
        Args:
            dataset (string): The name of preprocessed dataset
        """
        first_file = os.path.basename(data_path[0])
        last_file = os.path.basename(data_path[-1])
        with open(self.completion_file, 'a') as f:
            f.write(f"{dataset}_{subset} is all preprocessed from {first_file} to {last_file}.\n")
        
        print(f"Training complete! {len(data_path)} files from {first_file} to {last_file}. Created marker file at {self.completion_file}")

    # chunking specific
    def _chunk(self):
        return True
        
    def _check_if_complete():
        print("this must be implemented")
        return True
    
    def combine_datasets(self, lmdb_paths, merged_lmdb_path):
        """
        Combine multiple LMDB files into one.

        Args:
            lmdb_paths (list): List of LMDB file paths to combine.
            merged_lmdb_path (str): Path to save the merged LMDB.
        """

        # Delete merged path if already exists
        if os.path.exists(merged_lmdb_path):
            shutil.rmtree(merged_lmdb_path)

        # Open destination LMDB
        env_merged = lmdb.open(merged_lmdb_path, map_size=16492674416640)

        all_keys = []

        with env_merged.begin(write=True) as txn_merged:
            for path in lmdb_paths:
                env_source = lmdb.open(path, readonly=True, lock=False)
                with env_source.begin() as txn_source:
                    cursor = txn_source.cursor()
                    for key, value in cursor:
                        if key != b'__keys__':
                            txn_merged.put(key, value)
                            all_keys.append(key)
                        else:
                            keys_from_source = pickle.loads(value)
                            if isinstance(keys_from_source, list):
                                all_keys.extend([k.encode() if isinstance(k, str) else k for k in keys_from_source])
                            elif isinstance(keys_from_source, dict):
                                for sub_keys in keys_from_source.values():
                                    all_keys.extend([k.encode() if isinstance(k, str) else k for k in sub_keys])
                env_source.close()

            # Save merged __keys__
            txn_merged.put(b'__keys__', pickle.dumps(all_keys))

        env_merged.close()
        return True

    def merge_finetune_datasets(self, lmdb_info_list, merged_lmdb_path):
        if os.path.exists(merged_lmdb_path):
            shutil.rmtree(merged_lmdb_path)

        env_merged = lmdb.open(merged_lmdb_path, map_size=1649267441664)
        merged_keys = {'train': [], 'val': [], 'test': []}

        with env_merged.begin(write=True) as txn_merged:
            for subset, source_path in lmdb_info_list:
                env_source = lmdb.open(source_path, readonly=True, lock=False)
                with env_source.begin() as txn_source:
                    cursor = txn_source.cursor()
                    for key, value in cursor:
                        if key == b'__keys__':
                            continue
                        txn_merged.put(key, value)
                        decoded_key = key.decode() if isinstance(key, bytes) else key
                        merged_keys[subset].append(decoded_key)
                env_source.close()

            txn_merged.put(b'__keys__', pickle.dumps(merged_keys))

        env_merged.close()
        return True
        
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

#if __name__ == '__main__':
print("python file running")
start = time.time()

setup_seed(1)
data_path = params.data_path
dataset_name = params.dataset_name
completion_path = os.path.join(params.save_path_parent, f"{dataset_name}_{params.percent}_complete.txt")

print("set main")
preprocess_eeg = preprocessing_data(dataset = params.dataset_name, percent = params.percent, highpass = params.highpass, 
                                    lowpass = params.lowpass, notch_filter = params.notch_filter, 
                                    resample_rate = params.resample_rate, segment_len=params.segment_len, 
                                    data_path = data_path, save_path_parent = params.save_path_parent,
                                    resume = False, completion_path=completion_path)
file_dict = preprocess_eeg._get_sample_list_to_process(data_path, 
                                                               preprocess_eeg.dataset_class._file_ext(), 
                                                               params.percent, 
                                                               preprocess_eeg.dataset_class._get_split_dict())

chunk_dict = preprocess_eeg.chunk_dict(file_dict=file_dict, num_chunks=params.num_chunks)

print("check parallel")

for subset, chunk_paths in chunk_dict.items():
    for paths in chunk_paths:
        preprocess_eeg.run(dataset_name, subset, paths)

for subset in chunk_dict.keys():
    lmdb_path = (
        f"{params.save_path_parent}/{params.percent}_{dataset_name}/"
        f"{subset}_resample-{params.resample_rate}_highpass-{params.highpass}_lowpass-{params.lowpass}.lmdb"
    )
    env = lmdb.open(lmdb_path, map_size=16492674416640)
    with env.begin(write=True) as txn:
        cursor = txn.cursor()
        real_keys = []
        for key, _ in cursor:
            if key == b'__keys__':
                continue
            # bytes → str 로 변환 (선택사항)
            real_keys.append(key.decode() if isinstance(key, bytes) else key)
        # 진짜 존재하는 key 목록으로 __keys__ 덮어쓰기
        txn.put(b'__keys__', pickle.dumps(real_keys))
    env.close()

if preprocess_eeg.dataset_class._train_type() == 'finetune':
    preprocess_eeg.merge_finetune_datasets([
        ("train", f"{params.save_path_parent}/{params.percent}_{params.dataset_name}/train_resample-{params.resample_rate}_highpass-{params.highpass}_lowpass-{params.lowpass}.lmdb"),
        ("val", f"{params.save_path_parent}/{params.percent}_{params.dataset_name}/val_resample-{params.resample_rate}_highpass-{params.highpass}_lowpass-{params.lowpass}.lmdb"),
        ("test", f"{params.save_path_parent}/{params.percent}_{params.dataset_name}/test_resample-{params.resample_rate}_highpass-{params.highpass}_lowpass-{params.lowpass}.lmdb")
    ], f"{params.save_path_parent}/{params.percent}_{params.dataset_name}/merged_resample-{params.resample_rate}_highpass-{params.highpass}_lowpass-{params.lowpass}.lmdb")

elapsed = time.time() - start
print(f"Elapsed time: {elapsed:.2f} sec")

def _mark_completion(dataset, completion_path):
    """
    Create a file to mark training completion.
    
    Args:
        dataset (string): The name of preprocessed dataset
    """
    with open(completion_path, 'w') as f:
        f.write(f"{dataset} is all preprocessed")
    
    print(f"Training complete! Created marker file at {completion_path}")

_mark_completion(dataset_name, completion_path)
# nersc scratch 공간 늘리는 티켓 -> 

# if use chunk style parallelization