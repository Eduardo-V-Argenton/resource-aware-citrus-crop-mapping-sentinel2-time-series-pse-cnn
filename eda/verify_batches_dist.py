import pandas as pd
from os import listdir
import statistics

data = pd.read_csv('/mnt/SSD_SATA/dataset/dataset_index.csv')

batches_csv = listdir('paper_results/batch_tracking')

def count_classes(batches, label_map):
    class_1 = {}
    class_0 = {}

    for i, row in enumerate(batches.itertuples()):
        images = str(row.samples).split("|")

        c0 = 0
        c1 = 0

        for image in images:
            label = label_map.get(image)

            if label == 1:
                c1 += 1
            elif label == 0:
                c0 += 1
            else:
                print(f"Error: image {image} not found")

        class_0[i] = c0
        class_1[i] = c1

    return class_0, class_1
    
def stats(d):
    vals = list(d.values())
    return (
        sum(vals),
        sum(vals) / len(vals) if vals else 0,
        statistics.median(vals) if vals else 0
    )
    
label_map = dict(zip(data["name"], data["label_ia"]))

for batches in batches_csv:
    batch = pd.read_csv(f'paper_results/batch_tracking/{batches}')

    train_batch = batch[batch["phase"] == "train"]
    val_batch = batch[batch["phase"] == "val"]

    t_c0, t_c1 = count_classes(train_batch, label_map)
    v_c0, v_c1 = count_classes(val_batch, label_map)

    total_len = len(train_batch) + len(val_batch)

    print(f"====={batches}=====")

    for name, c0, c1 in [("train", t_c0, t_c1), ("val", v_c0, v_c1)]:
        s0, m0, med0 = stats(c0)
        s1, m1, med1 = stats(c1)

        print(f"{name} class_0 total: {s0}")
        print(f"{name} class_1 total: {s1}")
        print(f"{name} class_0 mean: {m0:.3f}")
        print(f"{name} class_1 mean: {m1:.3f}")
        print(f"{name} class_0 median: {med0}")
        print(f"{name} class_1 median: {med1}\n")

    print("\n")