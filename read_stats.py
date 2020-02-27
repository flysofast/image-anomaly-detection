import numpy as np
stats = np.load("eval/roc_benchmark.npy", allow_pickle=True).item()
min_cat = {"auc": 2}
max_cat = {"auc": 0}
min_all = {"auc": 2}
max_all = {"auc": 0}
for path, item in stats.items():
    print(path)
    if min_all["auc"] > item['mean_all']:
            min_all["path"] = path
            min_all["auc"] = item['mean_all']

    if max_all["auc"] < item['mean_all']:
        max_all["path"] = path
        max_all["auc"] = item['mean_all']

    for cat_perf in item["performance"]:
        print(f"{cat_perf['defect_type']}: {cat_perf['mean_auc']}")

        if min_cat["auc"] > cat_perf['mean_auc']:
            min_cat["path"] = path
            min_cat["cat"]=cat_perf['defect_type']
            min_cat["auc"] = cat_perf['mean_auc']
            
        if max_cat["auc"]<cat_perf['mean_auc']:
            max_cat["path"] = path
            max_cat["auc"] = cat_perf['mean_auc']
            max_cat["cat"] = cat_perf['defect_type']
            
    print(f"{path}: mean_all {item['mean_all']}")

print(f"Min cat: {min_cat['auc']} - {min_cat['cat']} - {min_cat['path']}")
print(f"Max cat: {max_cat['auc']} - {max_cat['cat']} - {max_cat['path']}")

print(f"Overall min: {min_all['auc']} - {min_all['path']}")
print(f"Overall max: {max_all['auc']} - {max_all['path']}")




