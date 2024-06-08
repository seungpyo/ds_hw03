import random
import sys
import os

debug = os.environ.get("DEBUG_DBSCAN")
if debug:
    print("DEBUG MODE")
    from tqdm import tqdm
input_file, n, eps, min_pts = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
)
db = []
label = dict()
with open(input_file, "r") as f:
    for line in f:
        id, x, y = line.strip().split("\t")
        db.append((int(id), float(x), float(y)))

if debug:
    db = sorted(db, key=lambda x: random.random())[:7000]


def range_query(db, metric, p, eps):
    return [q for q in db if metric(p, q) <= eps**2]


def metric(a, b):
    return (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


next_label = 0
for p in db:
    if label.get(p[0]) is not None:
        continue
    neighbors = range_query(db, metric, p, eps)
    if len(neighbors) < min_pts:
        label[p[0]] = -1
        continue
    if debug:
        print(f"new cluster: {next_label}")
        print(f"Total: {len(db)}")
        print(f"Left unclassified: {len([x for x in db if label.get(x[0]) is None])}")
    c = next_label
    next_label += 1
    label[p[0]] = c
    s = set(neighbors) - {p}
    while len(s) > 0:
        next_s = set()
        it = s if not debug else tqdm(s)
        for q in it:
            if label.get(q[0]) == -1:
                label[q[0]] = c
            if label.get(q[0]) is not None:
                continue
            neighbors = range_query(db, metric, q, eps)
            label[q[0]] = c
            if len(neighbors) < min_pts:
                continue
            next_s.update(neighbors)
        s = next_s


clusters = dict()
for v, l in label.items():
    if clusters.get(l) is None:
        clusters[l] = []
    clusters[l].append(v)

paths = []
for c, ids in clusters.items():
    if c == -1:
        continue
    p = f"{input_file.replace('.txt', '')}_cluster_{c}.txt"
    paths.append(p)
    with open(p, "w") as f:
        f.write("\n".join([str(id) for id in ids]))

if debug:
    print("outputs:")
    print(paths)
    from matplotlib import pyplot as plt

    colors = [
        (random.random(), random.random(), random.random()) for _ in range(next_label)
    ]
    for p in db:
        l = label.get(p[0])
        if l is None:
            plt.scatter(p[1], p[2], color=(0, 0, 0), s=0.2)
        elif l == -1:
            plt.scatter(p[1], p[2], color="b", s=0.2)
        else:
            plt.scatter(p[1], p[2], color=colors[label[p[0]]], s=0.2)
    plt.show()
