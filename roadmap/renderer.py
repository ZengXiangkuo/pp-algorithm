# -*- coding:utf-8 -*-


class Renderer(object):
    class LineSegment(object):
        def __init__(self, x, dx, max_y, next_node=None):
            self.x = x
            self.dx = dx
            self.max_y = max_y
            self.next = next_node

        def __lt__(self, other):
            return self.x < other.x or (self.x == other.x and self.dx < other.dx)

    def __init__(self):
        self.current_row = 0
        self.singular_points = []
        self.horizontal_lines = []
        self.activated_tab = []
        self.all_tab = []

        self.rows = 0
        self.cols = 0
        self.data = None

    def __reset(self, num):
        self.current_row = -1
        self.activated_tab = []
        self.all_tab = [[] for i in range(num)]

    def __insert(self, ring):
        num = len(ring)

        for i in range(num):
            x0, y0 = ring[i]
            x1, y1 = ring[(i + 1) % num]
            xe, ye = ring[(i + 2) % num]
            if int(y0) == int(y1):
                self.horizontal_lines.append([[x0, y0], [x1, y1]])
                continue
            if y0 < y1 and y1 > ye:
                self.singular_points.append([x1, y1])
            if y0 > y1:
                x0, x1 = x1, x0
                y0, y1 = y1, y0
            if y1 < 0 or y0 >= len(self.all_tab):
                continue
            dx = 0 if (int(y0) == int(y1)) else (x0 - x1) / (y0 - y1)
            if y0 < 0:
                x0 = x0 + (0 - y0) * dx
                y0 = 0
            min_y = int(round(y0))
            self.all_tab[min_y].append(Renderer.LineSegment(x0, dx, y1))

    def __update(self):
        self.current_row += 1
        self.activated_tab = [node for node in self.activated_tab if node.max_y != self.current_row]
        for node in self.activated_tab:
            node.x += node.dx
        self.activated_tab.extend(self.all_tab[self.current_row])
        self.activated_tab = sorted(self.activated_tab)

    def __fill(self, line_color, fill_color):
        if self.current_row < 0 or self.current_row >= self.rows:
            return

        num = len(self.activated_tab)

        for i in range(1, num, 2):
            start_col = int(self.activated_tab[i - 1].x)
            end_col = int(self.activated_tab[i].x)
            start_col = start_col if start_col >= 0 else 0
            end_col = end_col + 1 if end_col + 1 <= self.cols else self.cols

            for j in range(start_col + 1, end_col - 1):
                self.data[self.current_row][j] = fill_color

            if start_col < self.cols:
                self.data[self.current_row][start_col] = line_color

            if end_col >= 1:
                self.data[self.current_row][end_col - 1] = line_color

    def fill(self, data, rings, line_color, fill_color):

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

        max_y = int(max([p[1] for ring in rings for p in ring]))
        self.__reset(max_y + 1)
        for ring in rings:
            self.__insert(ring)

        self.__update()
        for i in range(len(self.all_tab) - 1):
            self.__fill(line_color, fill_color)
            self.__update()

        for p in self.singular_points:
            if 0 <= p[1] < self.rows and 0 <= p[0] < self.cols:
                self.data[p[1]][p[0]] = line_color

        for line in self.horizontal_lines:
            row = line[0][1]
            start_col, end_col = line[0][0], line[1][0]
            if start_col > end_col:
                start_col, end_col = end_col, start_col

            start_col = start_col if start_col >= 0 else 0
            end_col = end_col + 1 if end_col + 1 <= self.cols else self.cols
            if 0 <= row < self.rows:
                for j in range(start_col, end_col):
                    self.data[row][j] = line_color


def main():
    import matplotlib.pyplot as plt

    data = [[255] * 100 for _ in range(100)]

    ring1 = [[20, 20], [20, 80], [80, 80], [80, 20]]
    ring2 = [[30, 30], [50, 50], [70, 50], [70, 70], [50, 70], [40, 60]]

    renderer = Renderer()
    renderer.fill(data, [ring1, ring2], 0, 205)

    plt.figure()
    plt.axis('equal')
    plt.axis('off')
    plt.gray()
    plt.imshow(data)
    plt.show()


if __name__ == '__main__':
    main()
