from glob import glob
import os
import random
import numpy as np

def split_train_test(num_samples_each_class = 10):
    root = os.path.join("dataset","mvtec_anomaly_detection")
    category_paths = glob(os.path.join(root, "*/"))
    test_set = []
    for cat in category_paths:
        train_samples = [path + ",1" for path in glob(os.path.join(cat, "train", "**", "*.png"))]
        test_samples = [path + ",0" for path in glob(os.path.join(cat, "test", "**", "*.png"))]
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        test_set.extend(train_samples[:num_samples_each_class])
        test_set.extend(test_samples[:num_samples_each_class])

    with open(os.path.join(root, "train_test_split.txt"),"w") as textfile:
        for item in test_set:
            textfile.write(item + "\n")
