from src.preprocessing import load_and_filter
import matplotlib.pyplot as plt

# 1. Load data
epochs, raw = load_and_filter(1, 12)

# print(raw.info)

# print(raw.ch_names)
# print(raw.ch_names.index("C4"))
print(raw.annotations)
for ann in raw.annotations:
    if ann["description"] in ["T1", "T2"]:
        print("description : ", ann['description'] , 
        " | onset : ", ann['onset'], " | duration : ", ann['duration'])
print(epochs)
# 2. Visualize Signal
# raw.plot(n_channels=10, title="Filtered EEG (8-30Hz)")


# # 3. Visualize Power Spectral Density
# raw.compute_psd().plot()
# plt.show()