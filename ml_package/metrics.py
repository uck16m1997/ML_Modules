from ml_package import *


def calc_entropy(s):
    probs = []
    for c in s.value_counts():
        probs.append(c / len(s))
    entropy = 0
    for p in probs:
        entropy += -1 * p * np.log2(p)

    return entropy


def calc_info_gain(s, bin_masks):
    ent_all = calc_entropy(s)

    ent_binned = 0
    for m in bin_masks:
        p = sum(m) / len(m)
        ent = calc_entropy(s[m])
        ent_binned += p * ent

    inf_gain = ent_all - ent_binned
    return inf_gain
