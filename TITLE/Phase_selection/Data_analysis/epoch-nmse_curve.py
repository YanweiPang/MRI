import numpy as np
import matplotlib.pyplot as plt
import os
import json


random_file = "../results4ssimcurve/curvedata/run-random-70l_lightning_logs_version_0-tag-val_metrics_nmse.json"
zhang_file = "../results4ssimcurve/curvedata/run-evaluator_4gpu_test_lightning_logs_version_0-tag-val_metrics_nmse.json"
pineda_file = "../results4ssimcurve/curvedata/run-ss-ddqn-recon-org-70l_lightning_logs_version_0-tag-val_metrics_nmse.json"
title_file = "../results4ssimcurve/curvedata/run-recon_free_with_test_reward100_ss_trt_maskfused-70l_lightning_logs_version_0-tag-val_metrics_nmse.json"

save_dir = "../plots/nmse_curve"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


with open(random_file, "r") as f:
    random_data = json.load(f)
with open(zhang_file, "r") as f:
    zhang_data = json.load(f)
with open(pineda_file, "r") as f:
    pineda_data = json.load(f)
with open(title_file, "r") as f:
    title_data = json.load(f)

print(random_data)
print(zhang_data)
print(pineda_data)
print(title_data)

print(type(random_data))
random_data = np.array(random_data)
zhang_data = np.array(zhang_data)
pineda_data = np.array(pineda_data)
title_data = np.array(title_data)

random_x = random_data[:, 1] / 5480
zhang_x = zhang_data[:, 1] /1443
pineda_x = pineda_data[:, 1] / 5480
title_x = title_data[:, 1] / 5480

random_y = random_data[:, 2]
zhang_y = zhang_data[:, 2]
pineda_y = pineda_data[:, 2]
title_y = title_data[:, 2]


colors = [0.3694402, 0.5224523, 0.76407407, 0.89407407]

plt.figure()
plt.scatter(random_x, random_y, c='#D2691E', marker="+", label="Random")
plt.scatter(zhang_x, zhang_y, color="#9370DB", marker="s", label="Zhang")
plt.scatter(pineda_x, pineda_y, color="#1E90FF", marker="^", label="Pineda")
plt.scatter(title_x, title_y, color="#20B2AA", marker="*", label="TITLE")
plt.plot(random_x, random_y, c='#D2691E')
plt.plot(zhang_x, zhang_y, color="#9370DB")
plt.plot(pineda_x, pineda_y, color="#1E90FF")
plt.plot(title_x, title_y, color="#20B2AA")

plt.xlim(0, 30)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("NMSE")
plt.savefig(save_dir + "/nmse_curve.tiff", bbox_inches="tight")

plt.show()