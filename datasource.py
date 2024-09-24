import os
import sys
import random
import traceback
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import scipy

import mne
import neurokit2 as nk

import xml.etree.ElementTree as ET

import wfdb

import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset, TensorDataset

import neurokit2 as nk


minmax_scaler = MinMaxScaler()

# def get_rr_signal(y, hz, target_n_samp):
#     """ Get RR signal from raw ECG. """
#     _, info = nk.ecg_peaks(y, sampling_rate=hz, correct_artifacts=True, show=False)
#     r_peaks = info["ECG_R_Peaks"]
#     rr_signal = np.diff(r_peaks) / 2*hz
#     rr_signal = scipy.signal.resample(rr_signal, target_n_samp)
#     return rr_signal

# def get_edr(y, hz, target_n_samp):
#     """ Get ECG derived respiration """
#     _rpeaks, _ = nk.ecg_peaks(y, sampling_rate=hz, correct_artifacts=True, show=False)
#     ecg_rate = nk.signal_rate(_rpeaks, sampling_rate=hz, desired_length=len(_rpeaks))
#     rsp = nk.ecg_rsp(ecg_rate, sampling_rate=hz)
#     rsp_signal = scipy.signal.resample(rsp, target_n_samp)
#     rsp_signal = np.expand_dims(rsp_signal, axis=1)
#     rsp_signal = minmax_scaler.fit_transform(rsp_signal).flatten()
#     return rsp_signal

def get_derived_signals(y, hz, target_n_samp, rr=True, rsp=False):
    """ Get derived signals including RR, Rsp """
    ret_dict = {}
    if rr:
        _rpeaks, info = nk.ecg_peaks(y, sampling_rate=hz, correct_artifacts=True, show=False)
        r_peaks = info["ECG_R_Peaks"]
        rr_signal = np.diff(r_peaks) / 2*hz
        rr_signal = scipy.signal.resample(rr_signal, target_n_samp)
        rr_signal = np.expand_dims(rr_signal, axis=1)
        rr_signal = minmax_scaler.fit_transform(rr_signal).flatten()        
        ret_dict.update({
            "rr": rr_signal
        })
    
    if rsp:
        rpeaks_uncorrected = nk.signal_findpeaks(y)
        _, rpeaks_corrected = nk.signal_fixpeaks(
            rpeaks_uncorrected, sampling_rate=hz, iterative=False, method="Kubios", show=False)
        rate_corrected = nk.signal_rate(rpeaks_corrected, desired_length=len(y))
        # ecg_rate = nk.signal_rate(_rpeaks, sampling_rate=hz, desired_length=len(_rpeaks))
        rsp = nk.ecg_rsp(rate_corrected, sampling_rate=hz)
        rsp_signal = scipy.signal.resample(rsp, target_n_samp)
        rsp_signal = np.expand_dims(rsp_signal, axis=1)
        rsp_signal = minmax_scaler.fit_transform(rsp_signal).flatten()
        ret_dict.update({
            'rsp': rsp_signal
        })
    return ret_dict

def get_beats(y, hz, target_n_samp, n_beat=-1):
    """ ECG beats """
    beats_buf = []
    beats = nk.ecg_segment(y, rpeaks=None, sampling_rate=hz, show=False)
    for i in range(len(beats.keys())):
        if n_beat > -1 and i+1 > n_beat:
            break
        beat = beats[str(i+1)]['Signal']
        beat = scipy.signal.resample(beat, target_n_samp)
        beat = np.expand_dims(beat, axis=1)
        beat = minmax_scaler.fit_transform(beat).flatten()
        beats_buf.append(beat)

    "Duplicate last beat to meet n_beat"
    n_dup_beat = n_beat - len(beats_buf)
    for i in range(n_dup_beat):
        beats_buf.append(beats_buf[-1])
    
    beats_buf = np.stack(beats_buf, axis=1).T
    # print(f"beats: {beats_buf.shape}")
    return beats_buf


def load_edf_channel(edf_file, fs_target=100, ch_name=None, log=print):
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=False)
        log(f"channels: {raw.info.get('ch_names')}")
        ch_idx = -1        
        for cname in raw.info.get('ch_names'):
            ch_idx += 1
            if cname.upper().find(ch_name.upper()) > -1:
                break
        # else:
        #     raise Exception(f"No channel by name: {ch_name}")
        hz = mne.pick_info(raw.info, [ch_idx], verbose=False)['sfreq']
        hz = int(hz)
        raw.pick_channels([cname])
        recording = raw.get_data().flatten()
        log(f"channels: {raw.info.get('ch_names')}, search:{ch_name}, src_hz:{hz}")

        return scipy.signal.resample(recording, fs_target*len(recording) // int(hz))
    except:
        log(f"Error reading {edf_file}, caused by - {traceback.format_exc()}")
        return


def read_sleep_annot_xml(annot_xml):
    tree = ET.parse(annot_xml)
    root = tree.getroot()

    sleep_annot = []
    for se in root.findall('ScoredEvents'): 
        for evt in se:
            evt_meta = {}
            for item in evt:
                if item.tag == "EventType" and item.text != "Stages|Stages":
                    break
                if item.tag == "EventType":
                    continue
                val = item.text
                if item.tag == "EventConcept":
                    # <EventConcept>Wake|0</EventConcept>
                    # <EventConcept>Stage 1 sleep|1</EventConcept>
                    evt_meta['stage'] = int(val.split("|")[-1])
                if item.tag == "Start":
                    evt_meta["start"] = int(float(val))
                if item.tag == "Duration":
                    evt_meta["duration"] = int(float(val))
            else:
                # print(evt_meta)
                sleep_annot.append(evt_meta)
    return sleep_annot


def preprocess_ecg(recording, hz, diff=False, baseline_wader=False, fir_filter=None):
    # differentiated signal
    # 
    if diff:
        w = len(recording)    
        recording = np.diff(recording, n=1)
        if w != len(recording):
            recording = scipy.signal.resample(recording, w)

    # bandpass filter
    if fir_filter:
        filter = scipy.signal.firwin(400, [fir_filter[0], fir_filter[1]], pass_zero=False, fs=hz)
        recording = scipy.signal.convolve(recording, filter, mode='same')

    # remove baseline wander
    # 
    if baseline_wader:
        recording = hf.remove_baseline_wander(recording, sample_rate=hz, )

    # minmax scaling
    # 
    # recording = np.expand_dims(recording, axis=1)
    # recording = minmax_scaler.fit_transform(recording).flatten()
    return recording


def query_channel_names(
        data_path, n_subject=1, fs_target=64, seg_len=None, ch_name='ekg'
):
    count_recording = 0
    for f in os.listdir(data_path):
        if not f.endswith(".edf"):
            continue
        count_recording += 1
        if count_recording > n_subject:
            break
        recording = load_edf_channel(
                f"{data_path}/{f}", ch_name=ch_name, fs_target=fs_target)


class MesaDbCsv():
    def __init__(self, base_data_dir, data_subdir="edfs", hz=100, class_map=None, 
            hz_rr=5, filter_records=[], n_subjects=-1, is_ecg=True, is_rr_sig=False, 
            is_rsp=False, is_ecg_beats=False, n_classes=2, log=print) -> None:
        self.base_data_dir = base_data_dir
        self.data_dir = os.path.join(base_data_dir, data_subdir)
        self.hz = hz
        self.seg_len_sec = 30
        self.seg_dim = self.hz * self.seg_len_sec
        self.rr_seg_dim = hz_rr * self.seg_len_sec
        self.class_map = class_map
        self.n_classes = len(set(class_map.values()))
        self.filter_records = filter_records
        self.n_subjects = n_subjects
        self.is_rr_sig = is_rr_sig
        self.is_ecg_beats = is_ecg_beats
        self.is_rsp = is_rsp
        self.log = log
        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []
        self.log(
            f"Data base-dir:{base_data_dir}, data:{data_subdir}, "
            f"hz:{hz}, class_map:{class_map},")
        self._initialise()
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def _initialise(self):
        if self.n_subjects > 0:
            rec_names = []
            # randomly choose n recordings
            for f in os.listdir(self.data_dir):
                if not f.endswith("_annot.csv"):
                    continue
                # rec_name = f[:-4]
                rec_name = f.split("_")[0]
                rec_names.append(rec_name)
                if len(rec_names) > self.n_subjects:
                    self.filter_records = rec_names[:]
                    break
            self.log(f"Filter {len(self.filter_records)} records from {len(rec_names)}")

        count_file = 0
        
        for f in os.listdir(self.data_dir):
            if not f.endswith("_annot.csv"):
                continue
            # rec_name = f[:-4]
            rec_name = f.split("_")[0]
            if len(self.filter_records) > 0 and not rec_name in self.filter_records:
                continue

            # self.log(f"Loading {f} ...")

            self.record_names.append(rec_name)
            if self.record_wise_segments.get(rec_name) is None:
                self.record_wise_segments[rec_name] = []
            
            df_annot = pd.read_csv(os.path.join(self.data_dir, f"{rec_name}_annot.csv"))
            df_sig = pd.read_csv(os.path.join(self.data_dir, f"{rec_name}_sig.csv"))[f"{self.hz}hz"]

            clz_label_dist = {}
            seg_count = 0
            age = self.find_age(rec_name)
            for ind in df_annot.index:
                start = df_annot[f"{self.hz}hz_start"][ind]
                stage = df_annot['stage'][ind]
                is_noisy = False if df_annot['noisy'][ind]==0 else False
                label = self.class_map.get(stage)
                if label is None:
                    self.log(f"No label for annot '{stage}' in {f}")
                    continue   
                if is_noisy:
                    continue
                seg = df_sig[start:start+self.seg_dim]

                # print("assert: ", len(seg), self.seg_dim)
                # assert len(seg) == self.seg_dim
                if len(seg) != self.seg_dim:
                    continue

                # Valid segment, include them.
                # 
                seg_out = {
                    'ecg': seg,
                    'age': age,
                }
                if self.is_rr_sig or self.is_rsp:
                    try:
                        seg_out.update(
                            get_derived_signals(
                                seg, self.hz, self.rr_seg_dim, rr=self.is_rr_sig, rsp=self.is_rsp))
                    except:
                        print(traceback.format_exc())
                        continue   
                self.segments.append(seg_out)
                self.seg_labels.append(label)
                self.record_wise_segments[rec_name].append(len(self.segments)-1)
                
                if clz_label_dist.get(label) is None:
                    clz_label_dist[label] = 0
                clz_label_dist[label] += 1

                seg_count += 1
            self.log(
                f"[{f}] n_seg:{seg_count}, clz_lbl_dist:{clz_label_dist}")
            count_file += 1
        # sample distribution
        # 
        self.indexes = range(len(self.segments))
        _dist = np.unique(
            [self.seg_labels[i] for i in self.indexes], return_counts=True)
        self.log(f"Total files:{count_file}, n_seg:{len(self.segments)}, distribution:{_dist}")
        
    def find_age(self, rec_name):
        if not hasattr(self, 'df_age_info'):
            self.df_age_info = pd.read_csv(
                f"{os.path.expanduser('~')}/data/mesa/datasets/mesa-sleep-dataset-0.6.0.csv")
        rec_id = int(rec_name.split('-')[-1][:4])
        # age = self.df_age_info[self.df_age_info['mesaid']==rec_id].sleepage5c
        age = self.df_age_info[self.df_age_info['mesaid']==rec_id].sleepage5c.item()
        return 0 if age < 70 else 1


class MesaDb():
    def __init__(
            self, base_data_dir, data_subdir="edfs", annot_subdir="annotations-events-nsrr", 
            hz=128, seg_sec=30, class_map=None, log=print, hz_rr=5, # rr_seg_dim=100, 
            rr_min=0.2, rr_max=2, sig_modality="ekg", filter_records=[], 
            n_subjects=-1, is_rr_sig=False, is_rsp=False, is_ecg_beats=False) -> None:
        self.base_data_dir = base_data_dir
        self.data_dir = os.path.join(base_data_dir, data_subdir)
        self.annot_dir = os.path.join(base_data_dir, annot_subdir)
        self.hz = hz
        self.seg_sec = seg_sec
        self.seg_dim = seg_sec*hz
        self.class_map = class_map
        self.n_classes = len(set(class_map.values()))
        self.log = log
        self.sig_modality = sig_modality
        self.rr_seg_dim = hz_rr*seg_sec  # rr_seg_dim
        self.rr_min = rr_min
        self.rr_max = rr_max
        self.filter_records = filter_records
        self.n_subjects = n_subjects
        self.is_rr_sig = is_rr_sig
        self.is_ecg_beats = is_ecg_beats
        self.is_rsp = is_rsp
        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []
        self.log(
            f"Data base-dir:{base_data_dir}, data:{data_subdir}, annot:{annot_subdir}, "
            f"hz:{hz}, seg_sec:{seg_sec}, class_map:{class_map}, "
            f"n_classes:{self.n_classes}")
        self.scaler = MinMaxScaler()
        self._initialise()
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def _initialise(self):
        if self.n_subjects > 0:
            rec_names = []
            # randomly choose n recordings
            for f in os.listdir(self.data_dir):
                if not f.endswith(".edf"):
                    continue
                rec_name = f[:-4]
                rec_names.append(rec_name)
                if len(rec_names) > self.n_subjects:
                    self.filter_records = rec_names[:]
                    break
            # random_rec_names = set()
            # while len(set(random_rec_names)) < self.n_subjects:
            #     random_rec_names.add(
            #         rec_names[random.randint(0, len(rec_names))])
            # self.filter_records = list(random_rec_names)
            self.log(f"Filter {len(self.filter_records)} records from {len(rec_names)}")

        count_file = 0
        for f in os.listdir(self.data_dir):
            if not f.endswith(".edf"):
                    continue
            rec_name = f[:-4]
            if len(self.filter_records) > 0 and not rec_name in self.filter_records:
                continue
            data_file = f"{self.data_dir}/{f}"
            annot_file = f"{self.annot_dir}/{rec_name}-nsrr.xml"
  
            self.log(f"Loading {rec_name}...")
            sleep_annot = read_sleep_annot_xml(annot_file)
            
            recording = load_edf_channel(
                data_file, ch_name=self.sig_modality, fs_target=self.hz, log=self.log)
            
            if recording is None:
                continue

            self.record_names.append(rec_name)
            if self.record_wise_segments.get(rec_name) is None:
                self.record_wise_segments[rec_name] = []
                      
            """pre-processing recording level.
                Let's normalise in segment level instead of recording level.
            """
            recording = self.preprocess_recording(recording)
            age = self.find_age(rec_name)
            self.log(f"[{f[:-4]}] {len(sleep_annot)} events, age:{age}")            

            """segmentation"""
            seg_count = 0
            evt_count = 0
            annot_label_dist, clz_label_dist = {}, {}
            for dict_annot in sleep_annot:
                sleep_stage, start_sec, duration_sec = dict_annot['stage'], \
                    dict_annot['start'], dict_annot['duration']
                evt_count += 1
                label = self.class_map.get(sleep_stage)
                if annot_label_dist.get(sleep_stage) is None:
                    annot_label_dist[sleep_stage] = 0
                for i_epoch in range(duration_sec//self.seg_sec):
                    # Map to output clz label
                    annot_label_dist[sleep_stage] += 1                    
                    label = self.class_map.get(sleep_stage)
                    if label is None:
                        self.log(f"No label for annot '{sleep_stage}' in {f}")
                        continue                    

                    seg_start = (start_sec*self.hz) + (i_epoch*self.seg_dim)
                    seg = recording[seg_start:seg_start+self.seg_dim]

                    try:
                        seg = self.process_validate_segment(seg, f)
                    except:
                        print(traceback.format_exc())
                        continue                    
                    if seg is None:
                        continue

                    # include age
                    seg.update({
                        'age': age,
                    })
                    
                    # Valid segment, include them.
                    # 
                    self.segments.append(seg)
                    self.seg_labels.append(label)
                    self.record_wise_segments[rec_name].append(len(self.segments)-1)
                    
                    if clz_label_dist.get(label) is None:
                        clz_label_dist[label] = 0
                    clz_label_dist[label] += 1

                    seg_count += 1
                pass

            self.log(
                f"\tn_seg:{seg_count}, n_evt:{evt_count}, annot_dist:{annot_label_dist}, "
                f"clz_lbl_dist:{clz_label_dist}, remain:{len(recording)-(seg_start+self.seg_dim)}")
            count_file += 1
        # sample distribution
        # 
        self.indexes = range(len(self.segments))
        _dist = np.unique(
            [self.seg_labels[i] for i in self.indexes], return_counts=True)
        self.log(f"Total files:{count_file}, n_seg:{len(self.segments)}, distribution:{_dist}")

    def preprocess_recording(self, recording):
        """differentiated signal"""
        return nk.ecg_clean(recording, sampling_rate=self.hz, method="pantompkins1985")

    def process_validate_segment(self, seg, filename=None):
        # seg = np.expand_dims(seg, axis=1)
        # seg_z = self.scaler.fit_transform(seg)
        # seg_z = seg_z.flatten()
        seg_out = None
        ecg_cleaned = seg
        # ecg_cleaned = nk.ecg_clean(seg, sampling_rate=self.hz, method="pantompkins1985")
        quality = nk.ecg_quality(ecg_cleaned, method="zhao2018", sampling_rate=100)
        if "unacceptable".upper() == quality.upper():
            return seg_out

        ecg_cleaned = np.expand_dims(ecg_cleaned, axis=1)
        ecg_cleaned = minmax_scaler.fit_transform(ecg_cleaned).flatten()
        seg_out = get_derived_signals(
            ecg_cleaned, self.hz, self.rr_seg_dim, rr=self.is_rr_sig, rsp=self.is_rsp)        
        seg_out.update({
            'ecg': ecg_cleaned,
            'beats': get_beats(ecg_cleaned, self.hz, self.rr_seg_dim, n_beat=2) if self.is_ecg_beats else [],
        })
        
        # get beats
        # seg_out.update(
        #     get_beats(ecg_cleaned, self.hz, self.rr_seg_dim, n_beat=2)
        # )
        return seg_out

    def find_age(self, rec_name):
        if not hasattr(self, 'df_age_info'):
            self.df_age_info = pd.read_csv(
                f"{os.path.expanduser('~')}/data/mesa/datasets/mesa-sleep-dataset-0.6.0.csv")
        rec_id = int(rec_name.split('-')[-1][:4])
        # age = self.df_age_info[self.df_age_info['mesaid']==rec_id].sleepage5c
        age = self.df_age_info[self.df_age_info['mesaid']==rec_id].sleepage5c.item()
        return 0 if age < 70 else 1    


class PhysionetSlpDbCsv():
    def __init__(
            self, base_data_dir, hz=128, seg_sec=30, class_map=None, log=print, hz_rr=5, # rr_seg_dim=100, 
            rr_min=0.2, rr_max=2, sig_modality="ekg", filter_records=[], n_subjects=-1, 
            is_rr_sig=False, is_rsp=False, is_ecg_beats=False) -> None:
        self.data_dir = base_data_dir
        self.hz = hz
        self.seg_sec = seg_sec
        self.seg_dim = seg_sec * hz
        self.class_map = class_map
        self.n_classes = len(set(class_map.values()))
        self.log = log
        self.sig_modality = sig_modality
        self.rr_seg_dim = hz_rr*seg_sec  # rr_seg_dim
        # self.rr_seg_dim = rr_seg_dim
        self.rr_min = rr_min
        self.rr_max = rr_max
        self.filter_records = filter_records
        self.n_subjects = n_subjects
        self.is_rr_sig = is_rr_sig
        self.is_ecg_beats = is_ecg_beats
        self.is_rsp = is_rsp
        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []
        self.log(
            f"Data base-dir:{base_data_dir}, "
            f"hz:{hz}, seg_sec:{seg_sec}, class_map:{class_map}, "
            f"n_classes:{self.n_classes}")
        self.scaler = MinMaxScaler()
        self._initialise()
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def _initialise(self):
        if self.n_subjects > 0:
            rec_names = []
            # randomly choose n recordings
            for f in os.listdir(self.data_dir):
                if not f.endswith("_annot.csv"):
                    continue
                # rec_name = f[:-4]
                rec_name = f.split("_")[0]
                rec_names.append(rec_name)
                if len(rec_names) > self.n_subjects:
                    self.filter_records = rec_names[:]
                    break
            self.log(f"Filter {len(self.filter_records)} records from {len(rec_names)}")

        count_file = 0
        for f in os.listdir(self.data_dir):
            if not f.endswith("_annot.csv"):
                continue
            # rec_name = f[:-4]
            rec_name = f.split("_")[0]
            if len(self.filter_records) > 0 and not rec_name in self.filter_records:
                continue    
            
            self.log(f"Loading {rec_name}...")
            self.record_names.append(rec_name)
            if self.record_wise_segments.get(rec_name) is None:
                    self.record_wise_segments[rec_name] = []
                    
            
            df_annot = pd.read_csv(os.path.join(self.data_dir, f"{rec_name}_annot.csv"))
            df_sig = pd.read_csv(os.path.join(self.data_dir, f"{rec_name}_sig.csv"))[f"{self.hz}hz"]

            clz_label_dist = {}
            seg_count = 0
            for ind in df_annot.index:
                start = df_annot[f"{self.hz}hz_start"][ind]
                stage = df_annot['stage'][ind]
                is_noisy = False if df_annot['noisy'][ind]==0 else False
                label = self.class_map.get(stage)
                if label is None:
                    self.log(f"No label for annot '{stage}' in {f}")
                    continue   
                if is_noisy:
                    continue
                seg = df_sig[start:start+self.seg_dim]

                # Valid segment, include them.
                # 
                seg_out = {
                    'ecg': seg,
                }
                if self.is_rr_sig or self.is_rsp:
                    seg_out.update(
                        get_derived_signals(
                            seg, self.hz, self.rr_seg_dim, rr=self.is_rr_sig, rsp=self.is_rsp))
                self.segments.append(seg_out)
                self.seg_labels.append(label)
                self.record_wise_segments[rec_name].append(len(self.segments)-1)
                
                if clz_label_dist.get(label) is None:
                    clz_label_dist[label] = 0
                clz_label_dist[label] += 1

                seg_count += 1
            self.log(
                f"[{f}] n_seg:{seg_count}, clz_lbl_dist:{clz_label_dist}")
            count_file += 1
        # sample distribution
        # 
        self.indexes = range(len(self.segments))
        _dist = np.unique(
            [self.seg_labels[i] for i in self.indexes], return_counts=True)
        self.log(f"Total files:{count_file}, n_seg:{len(self.segments)}, distribution:{_dist}")
    

class PhysionetSlpDb():
    def __init__(
            self, base_data_dir, hz=128, seg_sec=30, class_map=None, log=print, hz_rr=5, # rr_seg_dim=100, 
            rr_min=0.2, rr_max=2, sig_modality="ekg", filter_records=[], n_subjects=-1, 
            is_rr_sig=False, is_rsp=False, is_ecg_beats=False) -> None:
        self.data_directory = base_data_dir
        self.hz = hz
        self.seg_sec = seg_sec
        self.seg_dim = seg_sec * hz
        self.class_map = class_map
        self.n_classes = len(set(class_map.values()))
        self.log = log
        self.sig_modality = sig_modality
        self.rr_seg_dim = hz_rr*seg_sec  # rr_seg_dim
        # self.rr_seg_dim = rr_seg_dim
        self.rr_min = rr_min
        self.rr_max = rr_max
        self.filter_records = filter_records
        self.n_subjects = n_subjects
        self.is_rr_sig = is_rr_sig
        self.is_ecg_beats = is_ecg_beats
        self.is_rsp = is_rsp
        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []
        self.log(
            f"Data base-dir:{base_data_dir}, "
            f"hz:{hz}, seg_sec:{seg_sec}, class_map:{class_map}, "
            f"n_classes:{self.n_classes}")
        self.scaler = MinMaxScaler()
        self._initialise()
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def _initialise(self):
        if self.n_subjects > 0:
            rec_names = []
            # randomly choose n recordings
            for f in os.listdir(self.data_directory):
                if not f.endswith(".hea"):
                    continue
                rec_name = f[:-4]
                rec_names.append(rec_name)
            random_rec_names = set()
            while len(set(random_rec_names)) < self.n_subjects:
                random_rec_names.add(
                    rec_names[random.randint(0, len(rec_names))])
            self.filter_records = list(random_rec_names)
            self.log(f"Filter {len(self.filter_records)} records from {len(rec_names)}")


        count_file = 0
        for f in os.listdir(self.data_directory):
            if not f.endswith(".hea"):
                continue
            rec_name = f[:-4]
            if len(self.filter_records) > 0 and not rec_name in self.filter_records:
                continue            
            
            self.log(f"Loading {rec_name}...")
            self.record_names.append(rec_name)
            if self.record_wise_segments.get(rec_name) is None:
                    self.record_wise_segments[rec_name] = []
                    
            signals, info = wfdb.rdsamp(f"{self.data_directory}/{rec_name}")
            
            # ECG signal exists - if yes, where?
            i_ecg = -1
            for i_sig, name_sig in enumerate(info['sig_name']):
                if name_sig.lower().find("ecg") > -1:
                    i_ecg = i_sig
                    break
            if i_ecg == -1:
                self.log(f"ERROR no ECG signal in record '{f}'")
                continue
            recording = signals[:, i_ecg]  # flatten vector

            # read annotation
            annot = wfdb.rdann(f"{self.data_directory}/{rec_name}", extension='st')

            recording = scipy.signal.resample(recording, self.hz*len(recording)//annot.fs)
            recording = self.preprocess_recording(recording)
            n_samples = len(recording)

            # segmentation
            # 
            seg_count = 0
            annot_label_dist, clz_label_dist = {}, {}
            for i_seg in range(len(annot.aux_note)):
                # aux_note is a list where first char is [W, 1, 2, 3, 4, R] 
                sleep_lbl = annot.aux_note[i_seg][:1]
                # count annot label distribution
                if annot_label_dist.get(sleep_lbl) is None:
                    annot_label_dist[sleep_lbl] = 0
                annot_label_dist[sleep_lbl] += 1

                # Map to output clz label
                label = self.class_map.get(sleep_lbl)
                if label is None:
                    self.log(f"No label for annot '{sleep_lbl}' in {f}")
                    continue
                if clz_label_dist.get(label) is None:
                    clz_label_dist[label] = 0
                clz_label_dist[label] += 1

                start = i_seg * self.seg_dim
                if start + self.seg_dim > n_samples:
                    self.log(
                        f"Remaining samples:{len(recording)-start}, "
                        f"annots:{len(annot.aux_note)-i_seg}")
                    break
                seg = recording[start : start + self.seg_dim]
                
                try:
                    seg = self.process_validate_segment(seg, f)
                except:
                    print(traceback.format_exc())
                    continue                    
                if seg is None:
                    continue                                 
                
                # Valid segment, include them.
                # 
                self.segments.append(seg)
                self.seg_labels.append(label)
                self.record_wise_segments[rec_name].append(len(self.segments)-1)
                
                if clz_label_dist.get(label) is None:
                    clz_label_dist[label] = 0
                clz_label_dist[label] += 1

                seg_count += 1
            pass

            self.log(
                f"... n_seg:{seg_count}, remain_samp:{len(recording)-(len(annot.aux_note)*self.seg_dim)}, "
                f"ignored_seg:{len(annot.aux_note)-seg_count}, annot_dist:{annot_label_dist}, "
                f"clz_lbl_dist:{clz_label_dist}")
            count_file += 1
        # sample distribution
        # 
        self.indexes = range(len(self.segments))
        _dist = np.unique(
            [self.seg_labels[i] for i in self.indexes], return_counts=True)
        self.log(f"Total files:{count_file}, n_seg:{len(self.segments)}, distribution:{_dist}")

    def preprocess_recording(self, recording):
        """differentiated signal"""
        return nk.ecg_clean(recording, sampling_rate=self.hz, method="pantompkins1985")

    def process_validate_segment(self, seg, filename=None):
        # seg = np.expand_dims(seg, axis=1)
        # seg_z = self.scaler.fit_transform(seg)
        # seg_z = seg_z.flatten()
        seg_out = None
        ecg_cleaned = seg
        # ecg_cleaned = nk.ecg_clean(seg, sampling_rate=self.hz, method="pantompkins1985")
        quality = nk.ecg_quality(ecg_cleaned, method="zhao2018", sampling_rate=100)
        if "unacceptable".upper() == quality.upper():
            return seg_out

        ecg_cleaned = np.expand_dims(ecg_cleaned, axis=1)
        ecg_cleaned = minmax_scaler.fit_transform(ecg_cleaned).flatten()
        seg_out = get_derived_signals(
            ecg_cleaned, self.hz, self.rr_seg_dim, rr=self.is_rr_sig, rsp=self.is_rsp)        
        seg_out.update({
            'ecg': ecg_cleaned,
            'beats': get_beats(ecg_cleaned, self.hz, self.rr_seg_dim, n_beat=2) if self.is_ecg_beats else [],
        })
        
        # get beats
        # seg_out.update(
        #     get_beats(ecg_cleaned, self.hz, self.rr_seg_dim, n_beat=2)
        # )
        return seg_out
    

class PartialDataset(Dataset):
    r"""Generate dataset from a parent dataset and indexes."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False, log=print
    ):
        r"""Instantiate dataset from parent dataset and indexes."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.log = log
        self.label_idx_dict = {}
        self._initialise()

    def _initialise(self):
        # label segregation
        # 
        for i_clz in range(self.memory_ds.n_classes):
            self.label_idx_dict[i_clz] = []
        for idx in self.indexes:
            self.label_idx_dict[self.memory_ds.seg_labels[idx]].append(idx)
        dist_str = [f"{i_clz}:{len(self.label_idx_dict[i_clz])}" for i_clz in range(self.memory_ds.n_classes)]
        self.log(f"label distribution: {dist_str}")

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        trainX = self.memory_ds.segments[ID]
        trainY = self.memory_ds.seg_labels[ID]

        tensor_dict = {}
        for k in trainX.keys():
            if k == 'age':
                tensor_dict[k] = trainX[k]
                continue
            _x = np.expand_dims(trainX[k], axis=1).T
            _x = Variable(torch.from_numpy(_x)).type(torch.FloatTensor)
            if torch.any(torch.isnan(_x)):
                _x = torch.nan_to_num(_x)
            tensor_dict[k] = _x
        tensor_dict['label'] = trainY

        # if self.memory_ds.is_rr_sig:
        #     trainX = np.stack([
        #         # trainX['ecg'],
        #         trainX['rr'],
        #         trainX['rsp']
        #         ], axis=1).T
        # else:
        #     trainX = np.expand_dims(trainX['ecg'], axis=1).T

        # X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        r"Segment in (1, n_samp) form, still need below line?"
        # X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        # Y_tensor = trainY
        # if torch.any(torch.isnan(X_tensor)):
        #     X_tensor = torch.nan_to_num(X_tensor)
        return tensor_dict
    

def main():
    datasource = MesaDb(
        base_data_dir=f"{os.path.expanduser('~')}/data/mesa/polysomnography", hz=100,
        # class_map={0:0, 1:1, 2:1, 3:1, 5:1}, 
        class_map={0:0, 1:1, 2:1, 3:1, 5:2},
        n_subjects=1, is_rr_sig=True)
    p_ds = PartialDataset(
        dataset=datasource, 
        seg_index=datasource.indexes[:100])
    for i in range(2):
        seg, lbl = p_ds[i]
        print(f"partial-ds, seg:{seg.shape}, label:{lbl}")               
        for ch in range(seg.shape[0]):
            plt.plot(range(seg.shape[-1]), seg[ch, :])
            # plt.ylim((0, 2)) 
            plt.title(f"ch:{ch}")
        plt.show()

            
if __name__ == "__main__":
    try:
        main()
    except:
        # traceback.print_exc(file=sys.stdout)
        print(traceback.format_exc())    