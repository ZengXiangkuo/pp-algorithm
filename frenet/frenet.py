# -*- coding:utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt


def bezier_fn(pa, pb, pc, pd, t):
    a = pa * (1 - t) ** 3
    b = 3 * pb * t * (1 - t) ** 2
    c = 3 * pc * t ** 2 * (1 - t)
    d = pd * t ** 3
    return a + b + c + d


def bezier_fn_d1(pa, pb, pc, pd, t):
    a = -3 * pa * (1 - t) ** 2
    b = 3 * pb * (1 - t) ** 2 - 6 * pb * t * (1 - t)
    c = 6 * pc * t * (1 - t) - 3 * pc * t ** 2
    d = 3 * pd * t ** 2
    return a + b + c + d


def bezier(points, num=100):
    assert (len(points) == 4)
    pa, pb, pc, pd = [np.array(p) for p in points]
    return [bezier_fn(pa, pb, pc, pd, i / num) for i in range(num)]


# reference points
refs = [[0, 0], [100, 0], [0, 100], [100, 100]]
refs = [np.array(p) for p in refs]

# normalize
normalize = lambda v: v / np.sqrt(v[0] * v[0] + v[1] * v[1])

# right_normal
right_normal = lambda v: np.array([v[1], -v[0]])

# position vector
r = lambda t: bezier_fn(*refs, t)

# normal vector
vn = lambda t: bezier_fn_d1(*refs, t)

# arrow
arrow = lambda t: [r(t), r(t) + 40 * normalize(vn(t))]

# create a bezier curve by two points and two normal vectors.
ps = bezier(refs)

# display it.
plt.figure()
plt.grid('on')

# beizer
plt.plot([v[0] for v in ps], [v[1] for v in ps], '-r', lw=2)
for i in range(41):
    arrow_ps = arrow(i / 40)
    plt.plot([v[0] for v in arrow_ps], [v[1] for v in arrow_ps], ':b', lw=0.5)

# move it 5
f5 = np.poly1d([5])
r_s = lambda t: r(t) + right_normal(normalize(vn(t))) * f5(t)
ps = [r_s(i / 100) for i in range(101)]
plt.plot([v[0] for v in ps], [v[1] for v in ps], '--g', lw=2)

plt.show()
