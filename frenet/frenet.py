# -*- coding:utf-8 -*-
import math

import numpy as np
import matplotlib.pyplot as plt


def spline(tt, yy):
    t, a, b, c, d = tt, [iy for iy in yy], [], [], []

    n = len(tt)
    h = np.diff(tt)

    # matrix a
    mat_a = np.zeros((n, n))
    mat_a[0, 0] = 1.0
    for i in range(n - 1):
        if i != (n - 2):
            mat_a[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
        mat_a[i + 1, i] = h[i]
        mat_a[i, i + 1] = h[i]
    mat_a[0, 1] = 0.0
    mat_a[n - 1, n - 2] = 0.0
    mat_a[n - 1, n - 1] = 1.0

    # matrix b
    mat_b = np.zeros((n, 1))
    for i in range(n - 2):
        mat_b[i + 1, 0] = 3.0 * ((a[i + 2] - a[i + 1]) / h[i + 1] - (a[i + 1] - a[i]) / h[i])
    mat_x = np.linalg.inv(np.mat(mat_a)) * np.mat(mat_b)
    c = [mat_x[i, 0] for i in range(n)]

    for i in range(n - 1):
        d.append((c[i + 1] - c[i]) / (3.0 * h[i]))
        tb = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        b.append(tb)

    return [t, a, b, c, d]


def spline_fn(pt, pa, pb, pc, pd, t):
    idx = 0

    while (idx < len(pt)) and (t > pt[idx]):
        idx += 1

    idx -= 1
    if idx >= len(pt) - 1:
        idx = len(pt) - 2
    elif idx < 0:
        idx = 0

    dt = t - pt[idx]
    return pa[idx] + pb[idx] * dt + pc[idx] * dt ** 2 + pd[idx] * dt ** 3


def bezier_fn(pa, pb, pc, pd, t):
    a = pa * (1 - t) ** 3
    b = 3 * pb * t * (1 - t) ** 2
    c = 3 * pc * t ** 2 * (1 - t)
    d = pd * t ** 3
    return a + b + c + d


class LaneData(object):

    def __init__(self):

        self.num = 0
        self.lane_xy = np.array([])
        self.lane_s = np.array([])
        self.lane_l = np.array([])

        self.lane_d = np.array([])
        self.lane_el = np.array([])
        self.lane_er = np.array([])

    def load(self, xys, ds):
        assert len(xys) == len(ds)

        self.num = len(xys)
        self.lane_xy = np.array(xys)
        self.lane_l = np.array(ds)
        self.lane_s = np.zeros((self.num, 1))

        self.lane_d = np.zeros((self.num, 2))

        self.lane_el = np.zeros((self.num, 2))
        self.lane_er = np.zeros((self.num, 2))

        for i in range(1, self.num):
            direct = self.lane_xy[i] - self.lane_xy[i - 1]

            self.lane_s[i] = self.lane_s[i - 1] + np.linalg.norm(direct)
            self.lane_d[i] = np.array(direct / np.linalg.norm(direct))

        self.lane_d[0] = self.lane_d[1]

        for i in range(self.num):
            l_direct = np.array([self.lane_d[i][1], -self.lane_d[i][0]])
            self.lane_el[i] = self.lane_xy[i] - self.lane_l[i] * l_direct
            self.lane_er[i] = self.lane_xy[i] + self.lane_l[i] * l_direct

    def frenet2xy(self, s, l):
        index = 0
        while index < self.num - 1 and self.lane_s[index + 1] < s:
            index += 1
        ds = s - self.lane_s[index]
        x_direct = self.lane_d[index]
        y_direct = np.array([-x_direct[1], x_direct[0]])
        return self.lane_xy[index] + ds * x_direct + l * y_direct


def quartic_polynomials(s0, sf):
    t0, y0, y0_d1, y0_d2 = s0
    tf, yf_d1, yf_d2 = sf
    a = [0] * 6
    a[0] = y0
    a[1] = y0_d1
    a[2] = 0.5 * y0_d2
    dt = tf - t0
    mat_a = [[3 * dt ** 2, 4 * dt ** 3, 5 * dt ** 4],
             [6 * dt, 12 * dt ** 2, 20 * dt ** 3]]
    mat_b = [[yf_d1 - a[1] - 2 * a[2] * dt],
             [yf_d2 - 2 * a[2]]]
    mat_x = np.linalg.inv(np.mat(mat_a)) * np.mat(mat_b)
    a[3] = mat_x[0, 0]
    a[4] = mat_x[1, 0]
    return a


def quint_polynomials(s0, sf):
    t0, y0, y0_d1, y0_d2 = s0
    tf, yf, yf_d1, yf_d2 = sf
    a = [0] * 6
    a[0] = y0
    a[1] = y0_d1
    a[2] = 0.5 * y0_d2
    dt = tf - t0
    mat_a = [[dt ** 3, dt ** 4, dt ** 5],
             [3 * dt ** 2, 4 * dt ** 3, 5 * dt ** 4],
             [6 * dt, 12 * dt ** 2, 20 * dt ** 3]]
    mat_b = [[yf - a[0] - a[1] * dt - a[2] * dt ** 2],
             [yf_d1 - a[1] - 2 * a[2] * dt],
             [yf_d2 - 2 * a[2]]]
    mat_x = np.linalg.inv(np.mat(mat_a)) * np.mat(mat_b)
    a[3] = mat_x[0, 0]
    a[4] = mat_x[1, 0]
    a[5] = mat_x[2, 0]
    return a


def compute_curve_params(params):
    src_s, src_l, src_yaw, dst_s, dst_l, dst_yaw = params
    sa = quint_polynomials([0, src_s, 100 * math.cos(src_yaw), 0],
                           [1, dst_s, 100 * math.cos(dst_yaw), 0])
    la = quint_polynomials([0, src_l, 100 * math.sin(src_yaw), 0],
                           [1, dst_l, 100 * math.sin(dst_yaw), 0])
    return sa[::-1] + la[::-1]


# create fake lane data by bezier curve function.
num = 100
refs = [[0, 0], [100, 0], [0, 100], [200, 100]]
refs = [np.array(p) for p in refs]

lane_data = LaneData()
lane_data.load([bezier_fn(*refs, i / num) for i in range(num)], [20] * num)

# display it.
plt.figure()
plt.axis('equal')

# draw fake lane
plt.plot([v[0] for v in lane_data.lane_xy], [v[1] for v in lane_data.lane_xy], '-k', lw=1)
plt.plot([v[0] for v in lane_data.lane_el], [v[1] for v in lane_data.lane_el], '-b', lw=1)
plt.plot([v[0] for v in lane_data.lane_er], [v[1] for v in lane_data.lane_er], '-b', lw=1)

# draw sample trajectory
for k in range(9):
    res = compute_curve_params([20, -5, np.pi / 12, 200, 5 * (k - 4), 0])
    trajectory_s = np.poly1d(res[0:6])
    trajectory_l = np.poly1d(res[6:12])

    points = []
    for i in range(lane_data.num):
        xy = lane_data.frenet2xy(trajectory_s(i / num), trajectory_l(i / num))
        points.append(xy)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.plot(xs, ys, '-g', lw=0.5)

plt.show()
