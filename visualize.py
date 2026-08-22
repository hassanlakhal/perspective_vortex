from src.preprocessing import load_and_filter

import matplotlib.pyplot as plt
import numpy as np
import mne

epochs, raw = load_and_filter(1, 8)

print(raw.info)
print(raw.ch_names)

print("\nAnnotations:")
print(raw.annotations)

# Print only T1 and T2 events
for ann in raw.annotations:
    if ann["description"] in ["T1", "T2"]:
        print(
            "description:", ann["description"],
            "| onset:", ann["onset"],
            "| duration:", ann["duration"]
        )

print("\nEpoch data shape:")
print(epochs.get_data().shape)

print("Number of epochs:", len(epochs))


epochs_t1 = epochs["T1"]
epochs_t2 = epochs["T2"]

print("\nT1 epochs:", len(epochs_t1))
print("T2 epochs:", len(epochs_t2))



evoked_t1 = epochs_t1.average()
evoked_t2 = epochs_t2.average()


baseline_interval = (-1.5, -0.5)
freqs = np.arange(8, 31, 1)

# TFR
power_t1 = mne.time_frequency.tfr_morlet(
    epochs_t1,
    freqs=freqs,
    n_cycles=freqs / 2,
    return_itc=False,
    average=False
)

power_t2 = mne.time_frequency.tfr_morlet(
    epochs_t2,
    freqs=freqs,
    n_cycles=freqs / 2,
    return_itc=False,
    average=False
)

# Average across epochs
power_t1_avg = power_t1.average()
power_t2_avg = power_t2.average()

# IMPORTANT:
# Apply baseline while the -0.2 -> 0 interval still exists
power_t1_avg.apply_baseline(
    baseline=baseline_interval,
    mode="percent"
)

power_t2_avg.apply_baseline(
    baseline=baseline_interval,
    mode="percent"
)

# NOW remove the baseline period from the displayed/data range
power_t1_avg.crop(tmin=0, tmax=4.1)
power_t2_avg.crop(tmin=0, tmax=4.1)

print("T1 after crop:")
print(power_t1_avg.times[0], power_t1_avg.times[-1])

print("T2 after crop:")
print(power_t2_avg.times[0], power_t2_avg.times[-1])

print(epochs.tmin, epochs.tmax)

power_t1_avg.plot(
    vlim=(-50, 50),
    cmap="RdBu_r",
    picks=["C3"],
    title="T1 - C3 - ERD/ERS"
)

plt.show()


power_t2_avg.plot(
    vlim=(-50, 50),
    cmap="RdBu_r",
    picks=["C3"],
    title="T2 - C3 - ERD/ERS"
)

plt.show()

# epochs_t1.compute_psd().plot()
# plt.show()

# epochs_t2.compute_psd().plot()
# plt.show()