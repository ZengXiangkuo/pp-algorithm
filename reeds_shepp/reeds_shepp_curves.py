# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def degree(*args):
    v = np.subtract(args[1], args[0]) if len(args) == 2 else args[0]
    return np.rad2deg(np.angle(v[0] + v[1] * 1j))


def norm(*args):
    v = np.subtract(args[1], args[0]) if len(args) == 2 else args[0]
    return np.linalg.norm(v)


def normalize(deg):
    while deg > 180:
        deg -= 360
    while deg <= -180:
        deg += 360
    return deg


def create_arc(center, radius, limit, is_left=True, interval=5):
    n = abs(int((limit[1] - limit[0]) / interval))
    angles = list(np.linspace(*limit, n + 1, endpoint=True))
    rad = -np.pi / 2 if is_left else np.pi / 2
    return [[
        center[0] + radius * np.cos(np.deg2rad(a) + rad),
        center[1] + radius * np.sin(np.deg2rad(a) + rad), a
    ] for a in angles]


class RsAction(object):
    def __init__(self,
                 start=None,
                 goal=None,
                 radius=0,
                 is_rotate=False,
                 is_left=True,
                 is_forward=True):

        self.is_rotate = is_rotate
        self.is_forward = is_forward
        self.is_left = is_left

        self.start_state = start if start else [0, 0, 0]
        self.goal_state = goal if goal else [0, 0, 0]
        self.radius = radius

    def move(self, dist):
        q0 = self.goal_state[:]
        q1 = [0, 0, 0]
        q1[2] = q0[2]
        q1[0] = q0[0] + dist * np.cos(np.deg2rad(q1[2]))
        q1[1] = q0[1] + dist * np.sin(np.deg2rad(q1[2]))
        return RsAction(q0, q1)

    def rotate(self, yaw, radius, is_left=True, is_forward=True):
        q0 = self.goal_state[:]
        q1 = [0, 0, yaw]

        theta = np.deg2rad(q0[2])
        normal = [-radius * np.sin(theta), radius * np.cos(theta)]

        if is_left:
            center = np.add([q0[0], q0[1]], normal)
            q1[0] = center[0] + radius * np.cos(np.deg2rad(yaw - 90))
            q1[1] = center[1] + radius * np.sin(np.deg2rad(yaw - 90))
        else:
            center = np.subtract([q0[0], q0[1]], normal)
            q1[0] = center[0] + radius * np.cos(np.deg2rad(yaw + 90))
            q1[1] = center[1] + radius * np.sin(np.deg2rad(yaw + 90))

        return RsAction(q0, q1, radius, True, is_left, is_forward)

    @staticmethod
    def scs(t, u, v):
        RsActions = [RsAction().move(t)]
        RsActions.append(RsActions[-1].rotate(*u))
        RsActions.append(RsActions[-1].move(v))
        points = []
        for a in RsActions:
            points.extend(a.to_points())
        return points

    @staticmethod
    def csc(t, u, v):
        RsActions = [RsAction().rotate(*t)]
        RsActions.append(RsActions[-1].move(u))
        RsActions.append(RsActions[-1].rotate(*v))
        points = []
        for a in RsActions:
            points.extend(a.to_points())
        return points

    @staticmethod
    def ccc(t, u, v):
        RsActions = [RsAction().rotate(*t)]
        RsActions.append(RsActions[-1].rotate(*u))
        RsActions.append(RsActions[-1].rotate(*v))
        points = []
        for a in RsActions:
            points.extend(a.to_points())
        return points

    def to_points(self):
        q0 = self.start_state[:]
        q1 = self.goal_state[:]

        if not self.is_rotate:
            return [q0, q1]

        radius = self.radius
        theta = np.deg2rad(q0[2])
        normal = [-radius * np.sin(theta), radius * np.cos(theta)]
        if self.is_left:
            center = np.add([q0[0], q0[1]], normal)
        else:
            center = np.subtract([q0[0], q0[1]], normal)

        limit = [q0[2], q1[2]]

        if self.is_left and self.is_forward:
            if limit[1] < limit[0]:
                limit[1] += 360
            return create_arc(center, self.radius, limit)
        elif self.is_left and not self.is_forward:
            if limit[1] > limit[0]:
                limit[1] -= 360
            return create_arc(center, self.radius, limit)
        elif not self.is_left and self.is_forward:
            if limit[1] < limit[0]:
                limit[1] += 360
            return create_arc(center, self.radius, limit, False)
        else:
            if limit[1] > limit[0]:
                limit[1] -= 360
            return create_arc(center, self.radius, limit, False)


class RsParams(object):
    def __init__(self):

        self.q0 = [0, 0, 0]
        self.q1 = [0, 0, 0]
        self.radius = 1

        self.loc_q0 = [0, 0, 0]
        self.loc_q1 = [0, 0, 0]
        self.l_trans = np.mat(np.eye(3, 3))
        self.g_trans = self.l_trans**-1

    def update_local(self):
        rad = np.deg2rad(self.q0[2])

        rotz = np.mat([[np.cos(rad), np.sin(rad), 0],
                       [-np.sin(rad), np.cos(rad), 0], [0, 0, 1]])

        trans = np.mat([[1, 0, -self.q0[0]], [0, 1, -self.q0[1]], [0, 0, 1]])

        xy = rotz * trans * np.mat([self.q1[0], self.q1[1], 1]).T

        self.l_trans = rotz * trans
        self.g_trans = self.l_trans**-1
        self.loc_q0 = [0, 0, 0]
        self.loc_q1 = [xy[0, 0], xy[1, 0], self.q1[2] - self.q0[2]]

    def to_local(self, q):
        xy = self.l_trans * np.mat([q[0], q[1], 1]).T
        return [xy[0, 0], xy[1, 0], normalize(q[2] - self.q0[2])]

    def to_global(self, q):
        xy = self.g_trans * np.mat([q[0], q[1], 1]).T
        return [xy[0, 0], xy[1, 0], normalize(q[2] + self.q0[2])]

    def to_global_points(self, qs):
        return [self.to_global(q) for q in qs]

    def scs(self):
        radius = self.radius
        x, y, theta = self.loc_q1

        # notice: parallel
        if theta == 0:
            return []

        xd = x - y / np.tan(np.deg2rad(theta))
        xd_arc = radius * np.tan(np.deg2rad(theta) / 2)
        yd = np.sqrt((x - xd)**2 + y**2)
        yd_arc = xd_arc

        paths = []
        # turning left
        t, u, v = xd - xd_arc, theta, yd - yd_arc
        if (self.loc_q1[2] > 0) != (self.loc_q1[1] > 0):
            v = -v
        paths.append(RsAction.scs(t, (u, radius, True, True), v))
        paths.append(RsAction.scs(t, (theta, radius, True, False), v))

        # turning right
        t, u, v = xd + xd_arc, theta, yd + yd_arc
        if (self.loc_q1[2] > 0) != (self.loc_q1[1] > 0):
            v = -v
        paths.append(RsAction.scs(t, (u, radius, False, True), v))
        paths.append(RsAction.scs(t, (u, radius, False, False), v))

        return paths

    def lsl(self, radius, t, u, v):
        paths = []
        for i in range(1 << 2):
            paths.append(
                RsAction.csc((t, radius, True, not (i & 0b1)), u,
                             (v, radius, True, not (i & 0b10))))
        return paths

    def rsr(self, radius, t, u, v):
        paths = []
        for i in range(1 << 2):
            paths.append(
                RsAction.csc((t, radius, False, not (i & 0b1)), u,
                             (v, radius, False, not (i & 0b10))))
        return paths

    def lsr(self, radius, t, u, v):
        paths = []
        for i in range(1 << 2):
            paths.append(
                RsAction.csc((t, radius, True, not (i & 0b1)), u,
                             (v, radius, False, not (i & 0b10))))
        return paths

    def rsl(self, radius, t, u, v):
        paths = []
        for i in range(1 << 2):
            paths.append(
                RsAction.csc((t, radius, False, not (i & 0b1)), u,
                             (v, radius, True, not (i & 0b10))))
        return paths

    def csc(self):
        radius = self.radius
        x, y, theta = self.loc_q1
        theta_rad = np.deg2rad(theta)
        normal = [-radius * np.sin(theta_rad), radius * np.cos(theta_rad)]

        center_sl = [0, radius]
        center_sr = [0, -radius]
        center_gl = np.add([x, y], normal)
        center_gr = np.subtract([x, y], normal)

        paths = []
        # LSL
        t = degree(center_sl, center_gl)
        u = norm(center_sl, center_gl)
        v = theta
        paths.extend(self.lsl(radius, t, u, v))
        paths.extend(self.lsl(radius, t + 180, -u, v))

        # RSR
        t = degree(center_sr, center_gr)
        u = norm(center_sr, center_gr)
        v = theta
        paths.extend(self.rsr(radius, t, u, v))
        paths.extend(self.rsr(radius, t + 180, -u, v))

        # LSR
        dist = norm(center_sl, center_gr)
        if dist > 2 * radius:
            dtheta = np.rad2deg(np.arcsin(2 * radius / dist))
            t = degree(center_sl, center_gr) + dtheta
            u = dist * np.cos(np.deg2rad(dtheta))
            v = theta
            paths.extend(self.lsr(radius, t, u, v))

            t = degree(center_sl, center_gr) - dtheta + 180
            u = -u
            paths.extend(self.lsr(radius, t, u, v))

        # RSL
        dist = norm(center_sr, center_gl)
        if dist > 2 * radius:
            dtheta = np.rad2deg(np.arcsin(2 * radius / dist))
            t = degree(center_sr, center_gl) - dtheta
            u = dist * np.cos(np.deg2rad(dtheta))
            v = theta
            paths.extend(self.rsl(radius, t, u, v))

            t = degree(center_sr, center_gl) + dtheta + 180
            u = -u
            paths.extend(self.rsl(radius, t, u, v))

        return paths

    def lrl(self, radius, t, u, v):
        paths = []
        for i in range(1 << 3):
            paths.append(
                RsAction.ccc((t, radius, True, not (i & 0b1)),
                             (u, radius, False, not (i & 0b10)),
                             (v, radius, True, not (i & 0b100))))

        return paths

    def rlr(self, radius, t, u, v):
        paths = []
        for i in range(1 << 3):
            paths.append(
                RsAction.ccc((t, radius, False, not (i & 0b1)),
                             (u, radius, True, not (i & 0b10)),
                             (v, radius, False, not (i & 0b100))))
        return paths

    def ccc(self):
        radius = self.radius
        x, y, theta = self.loc_q1
        theta_rad = np.deg2rad(theta)
        normal = [-radius * np.sin(theta_rad), radius * np.cos(theta_rad)]

        center_sl = [0, radius]
        center_sr = [0, -radius]
        center_gl = np.add([x, y], normal)
        center_gr = np.subtract([x, y], normal)

        paths = []

        # lrl
        dist = norm(center_sl, center_gl)
        if dist < 4 * radius:
            dtheta = np.rad2deg(np.arccos(dist / (4 * radius)))
            t = degree(center_sl, center_gl) + dtheta + 90
            u = t - 2 * dtheta + 180
            v = theta
            paths.extend(self.lrl(radius, t, u, v))
            t = degree(center_sl, center_gl) - dtheta + 90
            u = t + 2 * dtheta - 180
            v = theta
            paths.extend(self.lrl(radius, t, u, v))

        # rlr
        dist = norm(center_sr, center_gr)
        if dist < 4 * radius:
            dtheta = np.rad2deg(np.arccos(dist / (4 * radius)))
            t = degree(center_sr, center_gr) + dtheta - 90
            u = t - 2 * dtheta + 180
            v = theta
            paths.extend(self.rlr(radius, t, u, v))
            t = degree(center_sr, center_gr) - dtheta - 90
            u = t + 2 * dtheta - 180
            v = theta
            paths.extend(self.rlr(radius, t, u, v))

        return paths


def plot_paths(params, paths):
    min_length = np.inf
    min_path = None
    for path in paths:
        points = params.to_global_points(path)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, lw=0.5)

        length = sum([
            np.linalg.norm(np.subtract(points[i + 1][0:2], points[i][0:2]))
            for i in range(len(points) - 1)
        ])
        if length < min_length:
            min_length = length
            min_path = path

    if min_path:
        points = params.to_global_points(min_path)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, '-r', lw=2)

    plt.arrow(params.q0[0],
              params.q0[1],
              10 * np.cos(np.deg2rad(params.q0[2])),
              10 * np.sin(np.deg2rad(params.q0[2])),
              head_width=5)

    plt.arrow(params.q1[0],
              params.q1[1],
              10 * np.cos(np.deg2rad(params.q1[2])),
              10 * np.sin(np.deg2rad(params.q1[2])),
              head_width=5)
    plt.axis('equal')
    # plt.grid(True)


if __name__ == '__main__':

    params = RsParams()
    params.q0 = [0, 20, 15]
    params.q1 = [50, 50, 90]
    params.radius = 20
    params.update_local()

    all_paths = []

    plt.figure()

    # -----------------
    paths = params.scs()
    all_paths.extend(paths)

    plt.subplot(2, 2, 1)
    plt.title('SCS(%d)' % len(paths))
    plot_paths(params, paths)

    # -----------------
    paths = params.csc()
    all_paths.extend(paths)

    plt.subplot(2, 2, 2)
    plt.title(r'CSC(%d)' % len(paths))
    plot_paths(params, paths)

    # -----------------
    paths = params.ccc()
    all_paths.extend(paths)

    plt.subplot(2, 2, 3)
    plt.title(r'CCC(%d)' % len(paths))
    plot_paths(params, paths)

    # -----------------
    plt.subplot(2, 2, 4)

    plt.title(r'ALL(%d)' % len(all_paths))
    plot_paths(params, all_paths)

    plt.show()
