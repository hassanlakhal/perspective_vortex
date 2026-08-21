from src.preprocessing import load_and_filter

import matplotlib.pyplot as plt
import numpy as np
import mne

epochs, raw = load_and_filter(1, 7)

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


baseline_interval = (-0.2, 0)

evoked_t1_bc = evoked_t1.copy().apply_baseline(
    baseline_interval
)

evoked_t2_bc = evoked_t2.copy().apply_baseline(
    baseline_interval
)

freqs = np.arange(8, 31, 1)

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

print("\nT1 power shape:")
print(power_t1.data.shape)

print("\nT1 times:")
print(power_t1.times)

print("\nT1 frequencies:")
print(power_t1.freqs)

power_t1_avg = power_t1.average()
power_t2_avg = power_t2.average()

print("\nAverage T1 power shape:")
print(power_t1_avg.data.shape)

power_t1_avg.plot(
    baseline=baseline_interval,
    mode="percent",
    vlim=(-50, 50),
    cmap="RdBu_r",
    picks=["C3"]
)

# plt.suptitle("T1 - C3 Time-Frequency Power")
plt.show()


power_t2_avg.plot(
    baseline=baseline_interval,
    mode="percent",
    vlim=(-50, 50),
    cmap="RdBu_r",
    picks=["C4"]
)

# plt.suptitle("T2 - C4 Time-Frequency Power")
plt.show()

# epochs_t1.compute_psd().plot()
# plt.show()

# epochs_t2.compute_psd().plot()
# plt.show()