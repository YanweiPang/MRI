import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


parser = argparse.ArgumentParser()
parser.add_argument("--RF_num", type=int, default=70, help="How many RF to acquire.")
parser.add_argument("--baseline", type=str, default="evaluator")
args = parser.parse_args()

idx = args.RF_num
method = args.baseline

save_dir = "../plots/sampling_patterns/" + method
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

N = 1000
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1, 1, N)
vals[:, 1] = np.linspace(1, 0, N)
vals[:, 2] = np.linspace(1, 1, N)

vmax = 100
mycmap = LinearSegmentedColormap.from_list("mycmap", [(0/vmax, "white"), (2/vmax, "blue"), (100/vmax, "red")])
vds = cm.get_cmap("viridis")
newcolors = vds(np.linspace(0, 1, 1000))
white = np.array([250/256, 250/256, 250/256, 1])
newcolors = newcolors[::-1]
vdsw = ListedColormap(newcolors)

idx = args.RF_num
method = args.baseline
tjs_file = "path/to/trajectories.pkl"
with open(tjs_file, "rb") as f:
    tjs_list = pickle.load(f)

img = np.zeros((368, 101))
i = 0
for file_num, slice in tjs_list.items():
    for slice_num, tjs in slice.items():
        for step, smp in enumerate(tjs):
            img[int(smp), step:] += 1
        i += 1

img_cvs = img[::-1]

img_1d = np.sum(img, axis=1)
img_cvs_1d = np.sum(img_cvs, axis=1)
smt_1d = np.multiply(img_1d, img_cvs_1d)

smt = np.multiply(img, img_cvs)
print(np.sum(smt), np.sum(smt_1d))

img /= i
print(len(tjs_list))
img[169:199] = 1
fig = plt.figure(figsize=(3.3, 3))

ax = fig.add_subplot(111)

plt.xlabel("Acquisition Step")
plt.ylabel("Column")
plt.imshow(img, vmin=0, vmax=1, aspect="auto")
plt.savefig(save_dir + "/trajectories.png", bbox_inches="tight")
plt.show()