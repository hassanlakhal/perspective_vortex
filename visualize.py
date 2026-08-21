from src.preprocessing import load_and_filter
import matplotlib.pyplot as plt
import numpy as np
import mne
# 1. Load data
epochs, raw = load_and_filter(1, 11)

print(raw.info)

print(raw.ch_names)
# print(raw.ch_names.index("C4"))
print(raw.annotations)
for ann in raw.annotations:
    if ann["description"] in ["T1", "T2"]:
        print("description : ", ann['description'] , 
        " | onset : ", ann['onset'], " | duration : ", ann['duration'])
print(epochs.get_data().shape)
# print(epochs.ch_names.index("C4"))
times = np.arange(0, 4.1, 2)
print(len(epochs))


epochs_t1 = epochs["T1"]

epochs_t2 = epochs["T2"]



# print(epochs_t1)
# epochs_t1.average().plot_topomap(times, n_epochs=7)
# fig, axes = plt.subplots(7, len(times) + 1, figsize=(15, 9))

# for i in range(7):
#     evoked = epochs_t1[i].average()
#     evoked.plot_topomap(times, ch_type='eeg', axes=axes[i], show=False, vlim=(-50,50))
#     axes[i][0].set_title(f'Epoch-{i + 1}-')


evoked_t1 = epochs_t1.average()
evoked_t2 = epochs_t2.average()

inteval = (-0.2, 0)
bc_epochs_t1 = evoked_t1.apply_baseline(inteval)
bc_epochs_t2 = evoked_t2.apply_baseline(inteval)
# evoked_t1.plot_topomap(
#     times=times,
#     ch_type='eeg',
#     vlim=(-50,50)
# )

# evoked_t2.plot_topomap(
#     times=times,
#     ch_type='eeg',
#     vlim=(-50,50)
# )

# fig, axes = plt.subplots(2, 3, figsize=(12, 7))

# evoked_t1.plot_topomap(
#     times=times,
#     axes=axes[0],
#     colorbar=False,
#     vlim=(-45, 45),
#     show=False
# )

# evoked_t2.plot_topomap(
#     times=times,
#     axes=axes[1],
#     colorbar=False,
#     vlim=(-45, 45),
#     show=False
# )

# axes[0, 0].set_title("T1 mean - 0s")
# axes[0, 1].set_title("T1 mean - 2s")
# axes[0, 2].set_title("T1 mean - 4s")

# axes[1, 0].set_title("T2 mean - 0s")
# axes[1, 1].set_title("T2 mean - 2s")
# axes[1, 2].set_title("T2 mean - 4s")

# for i in range(0, 15):
# epochs[i].average().plot_topomap(times, ch_type='eeg')
# 2. Visualize Signal
# epochs[0].plot(n_channels=10, title="Filtered EEG (8-30Hz)")
# evoked_t1.compute_psd().plot()
# evoked_t2.compute_psd().plot()

bc_epochs_t1.plot()
plt.show()
bc_epochs_t2.plot()
plt.show()
# # # 3. Visualize Power Spectral Density
# epochs[0].compute_psd().plot()
# plt.tight_layout()