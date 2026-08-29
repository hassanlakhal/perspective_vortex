import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from src.preprocessing import load_and_filter
 
SUBJECT = 4
RUNS = [8]
L_FREQ, H_FREQ = 8.0, 30.0

def main():
    files = eegbci.load_data(SUBJECT, RUNS)
    raws = [read_raw_edf(f, preload=True) for f in files]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        on_missing="ignore",
    )
 
    print("=== BEFORE FILTERING ===")
    print(raw.info)
 
    fig1 = raw.plot(
        n_channels=10, duration=10, title="Raw EEG - BEFORE filtering",
        show=False,
    )
    fig1.savefig("raw_before_filter_timedomain.png", dpi=100)
    plt.close(fig1)
 
   
    psd_before = raw.compute_psd(fmax=80)
    fig2 = psd_before.plot(show=False)
    fig2.suptitle("PSD - BEFORE filtering (all frequencies present)")
    fig2.savefig("psd_before_filter.png", dpi=100)
    plt.close(fig2)
 
    epochs ,raw_filtered = load_and_filter(SUBJECT, RUNS)
    
    print("\n=== AFTER FILTERING ===")
    print(raw_filtered.info)
 
    fig3 = raw_filtered.plot(
        n_channels=10, duration=10,
        title=f"Raw EEG - AFTER filtering ({L_FREQ}-{H_FREQ}Hz)",
        show=False,
    )
    fig3.savefig("raw_after_filter_timedomain.png", dpi=100)
    plt.close(fig3)
 


    psd_after = raw_filtered.compute_psd(fmax=80)
    fig4 = psd_after.plot(show=False)
    fig4.suptitle(f"PSD - AFTER filtering ({L_FREQ}-{H_FREQ}Hz band isolated)")
    fig4.savefig("psd_after_filter.png", dpi=100)
    plt.close(fig4)
 
    fig5 =  epochs.plot(
        n_epochs=len(epochs),
        title="T1/T2 epochs",
        event_id=epochs.event_id,
        event_color={1: 'blue', 2: 'red'},
        show=False,
    )
    print(f"{epochs.event_id}")
    fig5.savefig("epochs_after_filter.png", dpi=100)
    plt.close(fig5)

    print("\nSaved 5 figures:")
    print(" - raw_before_filter_timedomain.png")
    print(" - psd_before_filter.png")
    print(" - raw_after_filter_timedomain.png")
    print(" - psd_after_filter.png")
    print(" - epochs_after_filter.png")
 
 
if __name__ == "__main__":
    main()