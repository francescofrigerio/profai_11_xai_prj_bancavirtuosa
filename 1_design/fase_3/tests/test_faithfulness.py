"""faithfulness.py: deletion curves rank a good map above a random order and a bad map below it."""
import dataclasses

import numpy as np

from helpers import tiny_cfg
from faithfulness import deletion_auc, deletion_curve, deletion_for_techniques

REGION = np.zeros((28, 28), dtype=bool)
REGION[8:20, 8:20] = True
IMAGE = np.zeros((28, 28))
IMAGE[8:20, 8:14] = 1.0                      # the model only looks at the LEFT half of the region


def predict(batch):
    """A model whose class-0 probability grows with the ink in the left half of the region."""
    ink = batch[:, 0, 8:20, 8:14].sum(axis=(1, 2))
    p = 1 / (1 + np.exp(-(ink - 36) / 6))
    return np.stack([p, 1 - p], axis=1)


def test_deletion_ranks_good_and_bad_maps_correctly():
    good = IMAGE.copy()                                              # attribution = where the ink really is
    bad = 1.0 - IMAGE                                                # the opposite
    c_good = deletion_curve(predict, IMAGE, good, REGION, 0, 10)
    c_bad = deletion_curve(predict, IMAGE, bad, REGION, 0, 10)
    rand = np.mean([deletion_auc(deletion_curve(predict, IMAGE, None, REGION, 0, 10, order="random",
                                                rng=np.random.RandomState(s))) for s in range(30)])
    assert len(c_good) == 11 and abs(c_good[0] - predict(IMAGE[None, None])[0, 0]) < 1e-12   # step 0 = untouched image
    assert deletion_auc(c_good) < rand < deletion_auc(c_bad)         # good < random < bad
    assert c_good[-1] < c_good[0]                                    # deleting the whole region hurts


def test_deletion_only_runs_when_the_flag_is_on():
    maps = {"good": IMAGE[None].copy()}
    args = (predict, maps, IMAGE[None], [0], REGION[None], 1)
    assert deletion_for_techniques(tiny_cfg(COMPUTE_FAITHFULNESS=False), *args) is None
    result = deletion_for_techniques(tiny_cfg(COMPUTE_FAITHFULNESS=True, DELETION_STEPS=10), *args)
    assert set(result) == {"good", "random"} and result["good"] < result["random"]
