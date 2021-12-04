from ml_package import *
from sklearn.decomposition import PCA
import math


def PCATransformer(x_train, x_test, groups):
    pca = PCA(n_components=0.8, svd_solver="full")
    x_train = x_train.copy()
    x_test = x_test.copy()
    for k, v in groups.items():
        tmp_train = x_train[v]
        tmp_test = x_test[v]
        x_train.drop(columns=v, inplace=True)
        x_test.drop(columns=v, inplace=True)
        pca.fit(tmp_train)
        for i in range(pca.n_components_):
            x_train[f"{k}_PC_{i}"] = pca.transform(tmp_train)[:, i]
            x_test[f"{k}_PC_{i}"] = pca.transform(tmp_test)[:, i]

    return (x_train, x_test)


def CenterTransformer(x_train, x_test, num_columns):
    x_train = x_train.copy()
    x_test = x_test.copy()
    for c in num_columns:
        mean = x_train[c].mean()
        x_train[c] = x_train[c] - mean
        x_test[c] = x_test[c] - mean

    return (x_train, x_test)


def CenterScaleTransform(x_train, x_test, scaler, pow_transformer, num_columns=None):
    if not num_columns:
        num_columns = x_train.select_dtypes(exclude=["object"]).columns
    x_train, x_test = CenterTransformer(x_train, x_test, num_columns)

    x_train[num_columns] = scaler.fit_transform(x_train[num_columns])
    x_test[num_columns] = scaler.transform(x_test[num_columns])

    x_train[num_columns] = pow_transformer.fit_transform(x_train[num_columns])
    x_test[num_columns] = pow_transformer.transform(x_test[num_columns])

    return x_train, x_test


def CorrelationClustering(X, columns=None, threshold=0.2):
    if not columns:
        col_details = data_prep.get_column_types(X)
        columns = col_details["Continious"] + col_details["Discrete"]
    corr_mat = X[columns].corr()
    distance_mat = 1 - np.abs(corr_mat)
    return AgglomerativeClustering(distance_mat, threshold)


def AgglomerativeClustering(distance_mat, threshold=0.2):
    groups = {}
    distance_mat.replace({0: 1}, inplace=True)
    values = np.sort((np.unique(distance_mat.values)))
    values = values[values < threshold]

    for v in values:
        cols = list(distance_mat.iloc[np.where(distance_mat == v)].columns)
        k = "_".join(cols)
        added_list = []
        keys = list(groups.keys())
        for key in keys:
            for c in cols:
                if c in key:
                    if len(added_list):
                        groups[key].update(groups[added_list[0]])
                        new_key = "_".join(groups[key])
                        groups[new_key] = groups.pop(key)
                        groups.pop(added_list[0])
                        added_list = [new_key]
                    else:
                        groups[key].update(cols)
                        new_key = "_".join(groups[key])
                        groups[new_key] = groups.pop(key)
                        added_list.append(new_key)

        if len(added_list) == 0:
            groups[k] = set()
            groups[k].update(cols)

    return groups


def InfoGainDiscretizer(
    x_train, x_test, y, split_count=None, iter_amount=100, regularization_rate=0.04
):
    x_repl = x_train.copy()
    x_repl_test = x_test.copy()

    x_repl["Target"] = y
    if type(x_repl) == type(pd.Series()):
        x_repl = pd.DataFrame(x_repl)
    for c in x_train.columns:
        tmp_sorted = x_repl.sort_values(by=[c])[[c, "Target"]]
        bins = optimize_infgain_bins(
            tmp_sorted[c],
            tmp_sorted["Target"],
            split_count=None,
            iter_amount=100,
            regularization_rate=0.001,
        )
        masks = [(x_repl[c] <= bins[0])]
        for i in range(len(bins) - 1):
            mask = (x_repl[c] > bins[i]) & (x_repl[c] <= bins[i + 1])
            masks.append(mask)
        masks.append(x_repl[c] > bins[-1])

        for i in range(len(masks)):
            x_repl.loc[masks[i], c] = i + 1.0

        masks = [(x_repl_test[c] <= bins[0])]
        for i in range(len(bins) - 1):
            mask = (x_repl_test[c] > bins[i]) & (x_repl_test[c] <= bins[i + 1])
            masks.append(mask)
        masks.append(x_repl_test[c] > bins[-1])

        for i in range(len(masks)):
            x_repl_test.loc[masks[i], c] = i + 1.0

    return x_repl.drop(columns=["Target"]), x_repl_test


def optimize_infgain_bins(
    x, y, split_count=None, iter_amount=100, regularization_rate=0.001
):
    if not split_count:
        split_count = list(range(1, 4))
    elif str(type(split_count)) != "list":
        split_count = list(split_count)
    res = {"Knots": [], "Infs": []}
    for s in split_count:
        knot, inf = inf_gain_binning_locked(x, y, s, iter_amount)
        res["Knots"].append(knot)
        res["Infs"].append(inf - s * regularization_rate)

    return res["Knots"][np.argmax(res["Infs"])]


def inf_gain_binning_locked(x, y, split_count=1, iter_amount=100):
    # Control if the number of splits is valid
    if split_count < 1 or split_count >= len(x.unique()):
        return [-1], -1
    # Create the increments
    inc_ind = len(x.unique()) // iter_amount + 1
    # Create the knots and masks
    knots_res = []
    masks_res = [[True] * len(x)]
    x_sort = x.unique()
    for i in range(split_count):
        # information gain and knots of current split
        inf_gain = []
        knots = []
        # initialize bins
        bins = [0, 0]
        # while bin index is less than
        while bins[0] + inc_ind < len(x_sort):
            bins[1] = (x_sort[bins[0]] + x_sort[bins[0] + inc_ind]) / 2
            # if knot is already found
            if bins[1] in knots_res:
                bins[0] += inc_ind
                continue
            # calculate the current splits effects
            tmp_knots = knots_res + [bins[1]]
            tmp_knots.sort()
            masks = [(x <= tmp_knots[0])]
            for i in range(len(tmp_knots) - 1):
                mask = (x > tmp_knots[i]) & (x <= tmp_knots[i + 1])
                masks.append(mask)
            masks.append(x > tmp_knots[-1])
            inf_gain.append(metrics.calc_info_gain(y, masks))
            # store the split number
            knots.append(bins[1])
            # update the bins
            bins[0] += inc_ind

        # add the new best knot and update masks
        knots_res.append(knots[np.argmax(inf_gain)])
        knots_res.sort()
        masks_res = [(x <= knots_res[0])]
        for i in range(len(knots_res) - 1):
            mask = (x > knots_res[i]) & (x <= knots_res[i + 1])
            masks_res.append(mask)
        masks_res.append(x > knots_res[-1])

    return knots_res, metrics.calc_info_gain(y, masks_res)


def inf_gain_binning_binary(x, y, split_count=1, iter_amount=100):
    # Control if the number of splits is valid
    if split_count < 1 or split_count > len(x.unique()):
        return "Wrong Number of Splits"
    # Create the increments
    inc_ind = len(x.unique()) // iter_amount + 1
    # Create the knots and masks
    knots_res = []
    masks_res = [[True] * len(x)]
    for i in range(split_count):
        # information gain and knots of current split
        inf_gain = []
        knots = []
        # Look through each bin for best information gain
        for m in masks_res:
            # create the bins
            x_masked = x[m]
            y_masked = y[m]
            # get sorted verison of the mask for bin calculations
            x_sort = x_masked.unique()
            # calculate first bin
            bins = [0, 0]
            # while bin index is less than
            while bins[0] + inc_ind < len(x_sort):
                bins[1] = (x_sort[bins[0]] + x_sort[bins[0] + inc_ind]) / 2
                # calculate the current splits effects
                mask = [(x_masked <= bins[1]), (x_masked > bins[1])]
                inf_gain.append(metrics.calc_info_gain(y_masked, mask))
                # store the split number
                knots.append(bins[1])
                # update the bins
                bins[0] += inc_ind

        # add the new best knot and update masks
        knots_res.append(knots[np.argmax(inf_gain)])
        knots_res.sort()
        masks_res = [(x <= knots_res[0])]
        for i in range(len(knots_res) - 1):
            mask = (x > knots_res[i]) & (x <= knots_res[i + 1])
            masks_res.append(mask)
        masks_res.append(x > knots_res[-1])

    return knots_res, metrics.calc_info_gain(y, masks_res)


### Warning Slow Code Below
def inf_gain_binning(x, y, split_count=1, iter_amount=100):
    # Control the number of splits is valid
    if split_count < 1 or split_count > len(x):
        return "Wrong Number of Splits"
    # Loop case for single split
    bin_ctrl = split_count == 1
    # Store Sorted array
    x_sort = sorted(x.unique())
    # Create the increments
    inc_ind = len(x_sort) // iter_amount + 1
    # Create the bins
    bins = [[0, (x_sort[0] + x_sort[inc_ind]) / 2]]
    for i in range(split_count - 1):
        bins.append(
            [
                bins[i][0] + inc_ind,
                (
                    (
                        x_sort[bins[i][0] + inc_ind]
                        + x_sort[bins[i][0] + inc_ind + inc_ind]
                    )
                    / 2
                ),
            ]
        )

    # Array to contain where seperation knots will be
    knots = []
    # Array to contain the information gain for bins
    inf_gains = []
    # Break if the last element of the bin is bigger than the max value of x
    while bins[-1][0] + inc_ind < len(x_sort):
        # If single split or first split knot hasn't yet reached to second split
        while bin_ctrl or bins[0][0] < bins[1][0]:

            # create bin masks
            masks = [(x <= bins[0][1])]
            for i in range(split_count - 1):
                mask = (x > bins[i][1]) & (x <= bins[i + 1][1])
                masks.append(mask)
            masks.append(x > bins[-1][1])

            # Calculate the information gain from bins and add it to the list
            inf_gains.append(metrics.calc_info_gain(y, masks))
            # Append the bins for that corresponds with the above info gain
            knots.append(pd.DataFrame(bins[:]))
            # Increment the first knot to create new bins
            bins[0][0] += inc_ind
            # Loop break for single bin case
            if bins[-1][0] + inc_ind >= len(x_sort):
                break
            bins[0][1] = (x_sort[bins[0][0]] + x_sort[bins[0][0] + inc_ind]) / 2
        if inf_gains[-1] < 0:
            break
        # Increase the upper bins and reset the lower bins
        for i in range(1, split_count):
            # Increase know for ith bin
            bins[i][0] += inc_ind
            # if i was the last knot go to the end
            if i == (split_count - 1):
                if bins[-1][0] + inc_ind >= len(x_sort):
                    break
            # if increased bin is same as the upper bin need to increase upper bin by continue
            elif bins[i][0] == bins[i + 1][0]:
                continue
            # all lower bins than i will be resetted
            bins[i][1] = (x_sort[bins[i][0]] + x_sort[bins[i][0] + inc_ind]) / 2
            bins[0] = [0, (x_sort[0] + x_sort[inc_ind]) / 2]
            for j in range(1, i):
                bins[j][0] = bins[j - 1][0] + inc_ind
                bins[j][1] = (x_sort[bins[j][0]] + x_sort[bins[j][0] + inc_ind]) / 2

            break

    return knots[np.argmax(inf_gains)][1], max(inf_gains)
