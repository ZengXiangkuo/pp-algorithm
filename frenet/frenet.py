# -*- coding:utf-8 -*-

import numpy as np
from scipy.optimize import fsolve
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


def bezier_fn_d2(pa, pb, pc, pd, t):
    a = 6 * pa * (1 - t)
    b = -6 * pb * (1 - t) - 6 * pb * (1 - 2 * t)
    c = 6 * pc * (1 - 2 * t) - 6 * pc * t
    d = 6 * pd * t
    return a + b + c + d


def bezier(points, num=100):
    assert (len(points) == 4)
    pa, pb, pc, pd = [np.array(p) for p in points]
    return [bezier_fn(pa, pb, pc, pd, i / num) for i in range(num)]


# reference points
refs = [[0, 0], [100, 0], [0, 100], [150, 100]]
refs = [np.array(p) for p in refs]


# normalize
def vec_normalize(v):
    return v / np.sqrt(v[0] * v[0] + v[1] * v[1])


# right_normal
def vec_right_normal(v):
    return np.array([v[1], -v[0]])


def vec_rate(v):
    return np.inf if v[0] == 0 else v[1] / v[0]


# position vector
def bezier_pos(t):
    return bezier_fn(*refs, t)


def bezier_pos_d1(t):
    return bezier_fn_d1(*refs, t)


def bezier_pos_d2(t):
    return bezier_fn_d2(*refs, t)


# normal vector
def bezier_normal(t):
    return vec_normalize(bezier_fn_d1(*refs, t))


def bezier_normal_d1(t):
    return vec_normalize(bezier_fn_d2(*refs, t))


# arrow
def bezier_arrow(t):
    return [bezier_pos(t), bezier_pos(t) + 40 * bezier_normal(t)]


def s_pos(fun, t):
    return bezier_pos(t) + vec_right_normal(bezier_normal(t)) * fun(t)


def s_pos_d1(fun, fun_d1, t):
    a = bezier_pos_d1(t)
    b = vec_right_normal(bezier_normal_d1(t)) * fun(t)
    c = vec_right_normal(bezier_normal(t)) * fun_d1(t)
    return a + b + c


# create a bezier curve by two points and two normal vectors.
ps = bezier(refs)

# display it.
plt.figure()
plt.grid('on')
plt.axis('equal')

# draw bezier curve
plt.plot([v[0] for v in ps], [v[1] for v in ps], '-r', lw=2)
for i in range(41):
    arrow_xy = bezier_arrow(i / 40)
    plt.plot([v[0] for v in arrow_xy], [v[1] for v in arrow_xy], ':b', lw=0.5)


def compute_curve_params(params):
    src_l = params[0]
    src_d = params[1]
    src_yaw = params[2]

    dst_l = params[3]
    dst_d = params[4]
    dst_yaw = params[5]

    def constraint(xx):
        assert len(xx) == 6
        fun = np.poly1d(xx)
        fun_d1 = fun.deriv(1)
        fun_d2 = fun.deriv(2)

        ret = [0] * 6

        # start pos
        ret[0] = fun(src_l) - src_d
        # start yaw
        ret[1] = vec_rate(s_pos_d1(fun, fun_d1, src_l)) - np.tan(src_yaw)
        # start acc
        ret[2] = fun_d2(src_l) - 0  # bezier_fn_d2(*refs, 0)
        # end pos
        ret[3] = fun(dst_l) - dst_d
        # end yaw
        ret[4] = vec_rate(s_pos_d1(fun, fun_d1, dst_l)) - vec_rate(bezier_normal(dst_l))
        # end acc
        ret[5] = fun_d2(dst_l) - 0  # bezier_fn_d2(*refs, 1)
        return ret

    return fsolve(constraint, [0] * 6)


for i in range(9):
    res = compute_curve_params([0, -20, 0, 1, (i - 4) * 5, 0])
    f5 = np.poly1d(res)
    ps = [s_pos(f5, i / 100) for i in range(101)]
    plt.plot([v[0] for v in ps], [v[1] for v in ps], '--g', lw=2)

plt.show()
