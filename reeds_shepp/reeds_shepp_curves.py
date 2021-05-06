# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Circle(object):

    def __init__(self, pos, radius, limit_deg=None):
        self.pos = pos
        self.radius = radius
        self.limit_deg = [0, 360] if limit_deg is None else limit_deg

    def xy(self):
        lb = int(self.limit_deg[0])
        rb = int(self.limit_deg[1])
        lb, rb = min(lb, rb), max(lb, rb)
        xs = [self.pos[0]+self.radius *
              np.cos(i/180*np.pi) for i in range(lb, rb+1)]
        ys = [self.pos[0]+self.radius *
              np.sin(i/180*np.pi) for i in range(lb, rb+1)]
        return xs, ys


class State(object):

    def __init__(self, pos, theta_deg):
        self.pos = pos
        self.theta_deg = theta_deg

    def get_left_circle(self, radius):
        rad = np.deg2rad(self.theta_deg)
        left_normal = [-radius*np.sin(rad), radius*np.cos(rad)]
        pos = [self.pos[0]+left_normal[0], self.pos[1]+left_normal[1]]
        return Circle(pos, radius)

    def get_right_circle(self, radius):
        rad = np.deg2rad(self.theta_deg)
        right_normal = [radius*np.sin(rad), -radius*np.cos(rad)]
        pos = [self.pos[0]+right_normal[0], self.pos[1]+right_normal[1]]
        return Circle(pos, radius)


if __name__ == '__main__':
    # todo reeds shepp curves generating
    plt.figure()

    plt.plot(*Circle([0, 0], 100).xy())

    plt.axis('equal')

    plt.show()
