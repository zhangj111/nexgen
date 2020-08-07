import sys

predict_file, target_file = sys.argv[1], sys.argv[2]
acc = []
with open(predict_file) as fpp, open(target_file) as fpt:
    for p, t in zip(fpp.readlines(), fpt.readlines()):
        if p.strip().split() == t.strip().split():
            acc.append(1)
        acc.append(0)

print(sum(acc)/len(acc))
