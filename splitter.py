from glob import glob
import os
import random
num_samples_each_class = 10
root = os.path.join("dataset","mvtec_anomaly_detection")
category_paths = glob(os.path.join(root, "*/"))
test_set = []
for cat in category_paths:
    train_samples = glob(os.path.join(cat, "train", "**", "*.png"))
    test_samples = glob(os.path.join(cat, "test", "**", "*.png"))
    test_set.extend(random.choices(train_samples,k=num_samples_each_class))
    test_set.extend(random.choices(test_samples,k=num_samples_each_class))

with open(os.path.join(root, "val_split.txt"),"w") as textfile:
    for item in test_set:
        textfile.write(item + "\n")
