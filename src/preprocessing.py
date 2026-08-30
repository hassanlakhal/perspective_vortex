import mne
from mne.datasets import eegbci
from moabb.datasets import BNCI2014_004
mne.set_log_level('ERROR')

def _load_physionet(subject, run, l_freq, h_freq):
    # Load PhysioNet data
    files = eegbci.load_data(subject, run)
    # raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
    raws = [mne.io.read_raw_edf(f) for f in files]
    raw = mne.io.concatenate_raws(raws)
    eegbci.standardize(raw)
    
    raw.load_data()
    # raw.pick(["C5", "C3", "C1", "Cz", "C2", "C4", "C6", "FC3", "FC4", "CP3", "CP4"])
    raw.pick(["C3", "C4"])
    # Filtering: 8-30Hz (Mandatory for Motor Imagery)
    raw.filter(8., 30., fir_design='firwin', verbose=False)
    
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    # Extract Events and Epochs
    events, event_id = mne.events_from_annotations(raw, dict(T1=1, T2=2), verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin=-2.5, tmax=4.5, 
                        picks='eeg', baseline=None, preload=True, verbose=False)
    return epochs, raw

def _load_bci2b(subject, l_freq=8.0, h_freq=30.0):
    
 
    dataset = BNCI2014_004()
    data = dataset.get_data(subjects=[subject])
 
    raws = []
    for session, runs in data[subject].items():
        for run_key, raw_obj in runs.items():
            raws.append(raw_obj)
 
    raw = mne.io.concatenate_raws(raws)
    raw.load_data()
    raw.pick_types(eeg=True)
 
    raw.filter(l_freq, h_freq, fir_design="firwin",
               skip_by_annotation="edge", verbose=False)
 
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep = {k: v for k, v in event_id.items()
            if k.lower() in ("left_hand", "right_hand")}
 
    epochs = mne.Epochs(
        raw, events, keep, tmin=-1.0, tmax=4.0,
        baseline=None, preload=True, verbose=False,
    )
    return epochs, raw



def load_and_filter(subject, runs 
                    ,l_freq=8.0, 
                    h_freq=30.0, source='physionet'):

    if source == 'physionet':
        print("---Dataset From physionet---")
        return _load_physionet(subject, runs, l_freq, h_freq)
    
    elif source == 'bci2b':
        print("---Dataset From bci2b---")
        return _load_bci2b(subject, l_freq, h_freq)