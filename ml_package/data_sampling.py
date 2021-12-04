from ml_package import *
from imblearn.under_sampling import RandomUnderSampler


def under_sample(X, y, sampling_strategy=0.5, random_state=42, random=False):
    # Under Sampling for more balanced target distribution
    if random:
        under = RandomUnderSampler(sampling_strategy=sampling_strategy)
    else:
        under = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=random_state
        )

    X, y = under.fit_resample(X, y)

    return X, y
