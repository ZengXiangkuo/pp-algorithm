# -*- coding:utf-8 -*-
import math

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def bezier_fn(pa, pb, pc, pd, t):
    a = pa * (1 - t) ** 3
    b = 3 * pb * t * (1 - t) ** 2
    c = 3 * pc * t ** 2 * (1 - t)
    d = pd * t ** 3
    return a + b + c + d


def vec_norm(v):
    return np.sqrt(v[0] * v[0] + v[1] * v[1])


# normalize
def vec_normalize(v):
    return v / np.sqrt(v[0] * v[0] + v[1] * v[1])


# right_normal
def vec_right_normal(v):
    return np.array([v[1], -v[0]])


def vec_rate(v):
    return np.inf if v[0] == 0 else v[1] / v[0]


def vec_add(v1, v2, rate1=1, rate2=1):
    return [v1[0] * rate1 + v2[0] * rate2, v1[1] * rate1 + v2[1] * rate2]


def compute_curve_params(params):
    src_s = params[0]
    src_l = params[1]
    src_yaw = params[2]

    dst_s = params[3]
    dst_l = params[4]
    dst_yaw = params[5]

    def constraint(xx):
        assert len(xx) == 12

        trajectory_s = np.poly1d(xx[0:6])
        trajectory_s_d1 = trajectory_s.deriv(1)
        trajectory_s_d2 = trajectory_s.deriv(2)

        trajectory_l = np.poly1d(xx[6:12])
        trajectory_l_d1 = trajectory_l.deriv(1)
        trajectory_l_d2 = trajectory_l.deriv(2)

        ret = [0] * 12

        ret[0] = trajectory_s(0) - src_s
        ret[1] = trajectory_s_d1(0) - 100
        ret[2] = trajectory_s_d2(0) - 0

        ret[3] = trajectory_s(1) - dst_s
        ret[4] = trajectory_s_d1(1) - 100
        ret[5] = trajectory_s_d2(1) - 0

        ret[6] = trajectory_l(0) - src_l
        ret[7] = trajectory_l_d1(0) - 100 * math.tan(src_yaw)
        ret[8] = trajectory_l_d2(0) - 0

        ret[9] = trajectory_l(1) - dst_l
        ret[10] = trajectory_l_d1(1) - 100 * math.tan(dst_yaw)
        ret[11] = trajectory_l_d2(1) - 0

        return ret

    return fsolve(constraint, np.array([0] * 12))


# create lane map data by bezier
num = 100
refs = [[0, 0], [100, 0], [0, 100], [150, 100]]
refs = [np.array(p) for p in refs]

# central axis
lane_xy = [bezier_fn(*refs, i / num) for i in range(num)]

# lane direction
lane_d = [0] * num

# lane s
lane_s = [0] * num

# lane l
lane_l = [100] * num

for i in range(1, num):
    v = vec_add(lane_xy[i], lane_xy[i - 1], 1, -1)
    lane_s[i] = lane_s[i - 1] + vec_norm(v)
    lane_d[i] = vec_normalize(np.array(v))
lane_d[0] = lane_d[1]

lane_e_l = [0] * num
lane_e_r = [0] * num
for i in range(num):
    y_direct = vec_right_normal(lane_d[i])
    lane_e_l[i] = vec_add(lane_xy[i], y_direct, 1, -20)
    lane_e_r[i] = vec_add(lane_xy[i], y_direct, 1, 20)


def frenet2xy(s, l):
    index = 0
    while index < num - 1 and lane_s[index + 1] < s:
        index += 1
    ds = s - lane_s[index]
    x_direct = lane_d[index]
    y_direct = vec_right_normal([-x_direct[0], -x_direct[1]])
    pos = vec_add(lane_xy[index], x_direct, 1, ds)
    return vec_add(pos, y_direct, 1, l)


# display it.
plt.figure()
plt.grid('on')
plt.axis('equal')

# draw bezier curve
plt.plot([v[0] for v in lane_xy], [v[1] for v in lane_xy], '-k', lw=1)
plt.plot([v[0] for v in lane_e_l], [v[1] for v in lane_e_l], '-b', lw=1)
plt.plot([v[0] for v in lane_e_r], [v[1] for v in lane_e_r], '-b', lw=1)

# draw sample trajectory
for k in range(10):
    res = compute_curve_params([20, -5, np.pi / 12, 200, 5 * (k - 5), 0])
    trajectory_s = np.poly1d(res[0:6])
    trajectory_l = np.poly1d(res[6:12])

    points = []
    for i in range(num):
        xy = frenet2xy(trajectory_s(i / num), trajectory_l(i / num))
        points.append(xy)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.plot(xs, ys, '-g', lw=0.5)

plt.show()
