from src.preprocessing import load_and_filter
import matplotlib.pyplot as plt
import numpy as np
# 1. Load data
epochs, raw = load_and_filter(1, 10)

print(raw.info)

# print(raw.ch_names)
# print(raw.ch_names.index("C4"))
print(raw.annotations)
for ann in raw.annotations:
    if ann["description"] in ["T1", "T2"]:
        print("description : ", ann['description'] , 
        " | onset : ", ann['onset'], " | duration : ", ann['duration'])
print(epochs.get_data().shape)
print(epochs.ch_names.index("C4"))
times = np.arange(0, 4.1, 2)
print(len(epochs))

fig, axes = plt.subplots(8, len(times) + 1, figsize=(15, 9))

for i in range(8):
    evoked = epochs[i].average()
    evoked.plot_topomap(times, ch_type='eeg', axes=axes[i], show=False)
    axes[i][0].set_title(f'Epoch-{i + 1}-')
# for i in range(0, 15):
# epochs[i].average().plot_topomap(times, ch_type='eeg')
# 2. Visualize Signal
# epochs[0].plot(n_channels=10, title="Filtered EEG (8-30Hz)")


# # # 3. Visualize Power Spectral Density
# epochs[0].compute_psd().plot()
plt.tight_layout()
plt.show()