# -*- coding:utf-8 -*-
from renderer import Renderer

import matplotlib.pyplot as plt
import numpy as np
import time


class Graph(object):
    class Node(object):
        def __init__(self, pos):
            self.pos = pos

    class Edge(object):
        def __init__(self, src, dst, pts):
            self.src = src
            self.dst = dst
            self.pts = pts

    def __init__(self):
        self.nodes = []
        self.edges = []


def create_simple_grid_map(sz):
    width, height = sz
    grid = np.ones((height, width)) * 205
    d = int(min(height, width) / 16)
    outer = [[0, 0], [int(width / 4 * 3), 0], [width - 1, int(height / 4)],
             [width - 1, height - 1], [0, height - 1]]
    inner_1 = [[int(width / 4) - d, int(height / 4) - d],
               [int(width / 4) + d, int(height / 4) - d],
               [int(width / 4) + d, int(height / 2) - d],
               [int(width / 2) + d, int(height / 2) - d],
               [int(width / 2) + d, int(height / 2) + d],
               [int(width / 4) + d, int(height / 2) + d],
               [int(width / 4) + d, int(height / 4 * 3) + d],
               [int(width / 4) - d, int(height / 4 * 3) + d]]

    inner_2 = [[int(width / 4 * 3) - d, int(height / 4) - d],
               [int(width / 4 * 3) + d, int(height / 4) - d],
               [int(width / 4 * 3) + d, int(height / 4 * 3) + d],
               [int(width / 4 * 3) - d, int(height / 2 * 3) + d],
               ]

    Renderer().fill(grid, [outer, inner_1, inner_2], 0, 255)

    return grid


def get_distance_map(img):
    m, n = img.shape
    img_bin = (img == 255) * 1
    img_dist = np.zeros((m, n))
    inf = max(m, n) ** 2

    # pre-computation for performance
    sqr_tab = [i ** 2 for i in range(max(m, n) + 1)]
    inv_tab = [0.5 / i if i != 0 else 0 for i in range(max(m, n) + 1)]
    sat_tab = [0 if i > m else i for i in range(2 * m + 1)]

    d = [0] * m
    for i in range(n):
        dt = m - 1
        for j in range(m - 1, -1, -1):
            dt = (dt + 1) & (0 if img_bin[j, i] == 0 else -1)
            d[j] = dt

        dt = m - 1
        for j in range(m):
            dt = dt + 1 - sat_tab[dt + 1 - d[j]]
            d[j] = dt
            img_dist[j, i] = sqr_tab[d[j]]

    f = [0] * (n + 1)
    z = [0] * (n + 1)
    v = [0] * (n + 1)
    for i in range(m):
        d = img_dist[i]

        v[0] = 0
        z[0] = -inf
        f[0] = d[0]
        k = 0
        for q in range(1, n):
            fq = d[q]
            f[q] = fq
            while True:
                p = v[k]
                s = (fq + sqr_tab[q] - d[p] - sqr_tab[p]) * inv_tab[q - p]
                if s > z[k]:
                    k += 1
                    v[k] = q
                    z[k] = s
                    z[k + 1] = inf
                    break
                k -= 1

        k = 0
        for q in range(n):
            while z[k + 1] < q:
                k += 1
            p = v[k]
            d[q] = np.sqrt(sqr_tab[abs(q - p)] + f[p])

    return img_dist


def get_road_map(img, min_dist=0, threshold=0.6):
    def principle_12(mat_33):
        total_val = mat_33.sum() - 1
        if not (2 <= total_val <= 6):
            return False

        arr = [*mat_33[0], mat_33[1, 2], *mat_33[2][::-1], mat_33[1, 0], mat_33[0, 0]]
        cnt = sum([1 if arr[idx] == 0 and arr[idx + 1] == 1 else 0 for idx in range(8)])
        if cnt != 1:
            return False

        return True

    def first_could_remove(mat_33):
        if not principle_12(mat_33):
            return False

        if mat_33[0, 1] * mat_33[1, 2] * mat_33[2, 1] != 0:
            return False

        if mat_33[1, 2] * mat_33[2, 1] * mat_33[2, 0] != 0:
            return False

        return True

    def second_could_remove(mat_33):
        if not principle_12(mat_33):
            return False

        if mat_33[0, 1] * mat_33[1, 2] * mat_33[2, 0] != 0:
            return False

        if mat_33[0, 1] * mat_33[2, 1] * mat_33[2, 0] != 0:
            return False

        return True

    m, n = img.shape
    img_buf = np.zeros((m, n))

    # laplace transform
    core = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            s = img[i - 1:i + 2, j - 1:j + 2]
            img_buf[i, j] = (s * core).sum()

    # binary transform
    min_val = img_buf.min()
    max_val = img_buf.max()
    img_buf = (img_buf - min_val) / (max_val - min_val)
    img_buf = ((img_buf > threshold) * (img > min_dist)) * 1

    # thinning 1
    img_copy = np.copy(img_buf)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if img_copy[i, j] == 0:
                continue
            if first_could_remove(img_copy[i - 1:i + 2, j - 1:j + 2]):
                img_buf[i, j] = 0

    # thinning 2
    img_copy = np.copy(img_buf)
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if img_copy[i, j] == 0:
                continue
            if second_could_remove(img_copy[i - 1:i + 2, j - 1:j + 2]):
                img_buf[i, j] = 0

    return img_buf


def is_end_point(mat_33):
    if mat_33[1, 1] != 1:
        return False

    total = mat_33.sum()
    if total == 1:
        return False
    cnt = 0
    arr = [*mat_33[0], mat_33[1, 2], *mat_33[2][::-1], mat_33[1, 0], mat_33[0, 0]]
    for i in range(8):
        if arr[i] == 1 and arr[i + 1] == 1:
            cnt += 1

    if total - cnt - 1 == 2:
        return False

    return True


def pos2index(cols, pos):
    return pos[0] * cols + pos[1]


def index2pos(cols, index):
    row = int(index / cols)
    col = index - row * cols
    return [row, col]


def get_graph(img):
    m, n = img.shape

    all_nodes = []
    all_edges = []
    node_set = []

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if is_end_point(img[i - 1:i + 2, j - 1:j + 2]):
                all_nodes.append([i, j])
                node_set.append(pos2index(n, [i, j]))

    motions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]

    visited = [*node_set]
    for node in all_nodes:
        stack = [node]
        two_point_tab = []

        while stack:
            cur = stack[-1]
            hash_next = False
            # print(node, stack, cur)
            for motion in motions:
                tmp = [cur[0] + motion[0], cur[1] + motion[1]]

                if not (0 < tmp[0] < m and 0 < tmp[1] < n):
                    continue

                if img[tmp[0], tmp[1]] == 0:
                    continue

                index = pos2index(n, tmp)

                if index in two_point_tab:
                    continue

                if index in node_set and index != pos2index(n, stack[0]):
                    two_point_tab.append(index)
                    all_edges.append([*stack, tmp])
                    stack = stack[0:1]
                    hash_next = True
                    break

                if index in visited:
                    continue

                visited.append(index)
                stack.append(tmp)
                hash_next = True
                break

            if not hash_next:
                if len(stack) > 1:
                    all_edges.append(stack)
                break

    g = Graph()
    g.nodes = [Graph.Node(node) for node in all_nodes]

    for pts in all_edges:
        src = node_set.index(pos2index(n, pts[0]))
        dst = node_set.index(pos2index(n, pts[-1]))
        g.edges.append(g.Edge(src, dst, pts))

    return g


grid_map = create_simple_grid_map((500, 500))

start = time.time()
dist_map = get_distance_map(grid_map)
end = time.time()
print('It took %0.3f ms to converting grid map to distance map completely.' % ((end - start) * 1000))

start = time.time()
road_map = get_road_map(dist_map, 10, 0.5)
end = time.time()
print('It took %0.3f ms to converting distance map to road map completely.' % ((end - start) * 1000))

start = time.time()
graph = get_graph(road_map)
end = time.time()
print('It took %0.3f ms to converting road map to graph completely.' % ((end - start) * 1000))

plt.figure()

plt.subplot(1, 3, 1)
plt.title('origin grid map')
plt.axis('equal')
plt.axis('off')
plt.gray()
plt.imshow(grid_map)

plt.subplot(1, 3, 2)
plt.title('distance map')
plt.axis('equal')
plt.axis('off')
plt.gray()
plt.imshow(dist_map)

plt.subplot(1, 3, 3)
plt.title('road map and graph diagram')
plt.axis('equal')
plt.axis('off')
plt.gray()

plt.imshow(road_map)

for edge in graph.edges:
    xs = [p[1] for p in edge.pts]
    ys = [p[0] for p in edge.pts]
    plt.plot(xs, ys)

xs = [p.pos[1] for p in graph.nodes]
ys = [p.pos[0] for p in graph.nodes]
plt.plot(xs, ys, '.b')

plt.show()
