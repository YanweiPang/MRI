import pickle as pickle

tjs_trainset_dir = "path/to/trainset_trajectories/trajectories.pkl"
tjs_valset_dir = "path/to/valset_trajectories/trajectories.pkl"
tjs_testset_dir = "path/to/testset_trajectories/trajectories.pkl"

tjs_merge_dir = "path/to/merged_trajectories/trajectories.pkl"

with open(tjs_trainset_dir, "rb") as f_tjs_train:
    tjs_train = pickle.load(f_tjs_train)
    print("trainset len:", len(tjs_train))

with open(tjs_valset_dir, "rb") as f_tjs_val:
    tjs_val = pickle.load(f_tjs_val)
    print("valset len:", len(tjs_val))

with open(tjs_testset_dir, "rb") as f_tjs_test:
    tjs_test = pickle.load(f_tjs_test)
    print("tjs_test len:", len(tjs_test))


tjs_merge = {}
tjs_merge.update(tjs_train)
tjs_merge.update(tjs_val)
tjs_merge.update(tjs_test)

with open(tjs_merge_dir, "wb") as f_tjs_merge:
    pickle.dump(tjs_merge, f_tjs_merge)

with open(tjs_merge_dir, "rb") as f_tjs_merged:
    tjs_merged = pickle.load(f_tjs_merged)
    print("tjs_merged len:", len(tjs_merged))