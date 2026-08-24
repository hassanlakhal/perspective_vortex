import mne
from mne.datasets import eegbci
mne.set_log_level('ERROR')

def load_and_filter(subject, run):
    # Load PhysioNet data
    files = eegbci.load_data(subject, run)
    # raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
    raws = [mne.io.read_raw_edf(f) for f in files]
    raw = mne.io.concatenate_raws(raws)
    eegbci.standardize(raw)
    
    raw.load_data()
    raw.pick(["C5", "C3", "C1", "Cz", "C2", "C4", "C6", "FC3", "FC4", "CP3", "CP4"])
    # Filtering: 8-30Hz (Mandatory for Motor Imagery)
    raw.filter(8., 30., fir_design='firwin', verbose=False)
    
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    # Extract Events and Epochs
    events, event_id = mne.events_from_annotations(raw, dict(T1=1, T2=2), verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin=-2.5, tmax=4.5, 
                        picks='eeg', baseline=None, preload=True, verbose=False)
    return epochs, raw