import random
from operator import xor
import itertools


def gt_or_fun(a, b):
    return a > .5 or b > .5


def gt_and_fun(a, b):
    return a > .5 and b > .5


def gt_xor_fun(a, b):
    return xor(bool(a > .5), bool(b > .5))


def equ_fun(a, b):
    return abs(a - b) < .2


def lt_gt_or_fun(a, b):
    return a < .5 or b > .5


def lt_gt_and_fun(a, b):
    return a < .5 and b > .5


def lt_gt_xor_fun(a, b):
    return xor(bool(a < .5), bool(b > .5))


def lt_or_fun(a, b):
    return a < .5 or b < .5


def lt_and_fun(a, b):
    return a < .5 and b < .5


def lt_xor_fun(a, b):
    return xor(bool(a < .5), bool(b < .5))


def gt_lt_or_fun(a, b):
    return a > .5 or b < .5


def gt_lt_and_fun(a, b):
    return a > .5 and b < .5


def gt_lt_xor_fun(a, b):
    return xor(bool(a > .5), bool(b < .5))


sensitive_features = ["race", "gender"]
sensitive_features_values = [[0, 1], [0, 1]]
other_features = ["feature1", "feature2"]
target_feature = "target"
operations = [gt_or_fun, gt_and_fun, lt_gt_or_fun, lt_gt_and_fun]
samples_per_group = [1000, 1000, 1000, 100]

print(",".join(sensitive_features + other_features + [target_feature]))
count = 0
for combo in itertools.product(*sensitive_features_values):
    for i in range(samples_per_group[count]):
        features = [random.random() for _ in other_features]
        target = int(operations[count](features[0], features[1]))
        print(",".join(map(str, [combo[0], combo[1], features[0], features[1], target])))
    count += 1
