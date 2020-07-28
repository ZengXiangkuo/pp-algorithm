# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def bezier_fn(pa, pb, pc, pd, t):
    return pa * (1 - t) ** 3 + 3 * pb * t * (1 - t) ** 2 + 3 * pc * t ** 2 * (1 - t) + pd * t ** 3


def bezier(points, num=100):
    assert (len(points) == 4)
    pa, pb, pc, pd = [np.array(p) for p in points]
    return [bezier_fn(pa, pb, pc, pd, i / num) for i in range(num)]


def bezier_curve(p1, p2, v1, v2):
    points = [p1, [p1[0] + v1[0], p1[1] + v1[1]], [p2[0] + v2[0], p2[1] + v2[1]], p2]
    return bezier(points)


# create a bezier curve by two points and two normal vectors.
ps = bezier_curve([0, 0], [100, 0], [1, 1], [-2, 2])

# display it.
plt.figure()
plt.grid('on')
plt.plot([v[0] for v in ps], [v[1] for v in ps])
plt.show()
