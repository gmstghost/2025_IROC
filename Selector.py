import numpy as np
import logging



class Selector:
    def __init__(self, stack=None):
        self.stack = stack


    def _get_center(self, rect):
        x, y, w, h = rect
        return (x + w / 2, y + h / 2)


    def _get_distance(self, pos1, pos2):
        x_valid = pos1[0] is not None and pos2[0] is not None
        y_valid = pos1[1] is not None and pos2[1] is not None

        if x_valid and y_valid:
            return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) ** 0.5
        if x_valid:
            return abs(pos1[0] - pos2[0])
        if y_valid:
            return abs(pos1[1] - pos2[1])
        logging.error("SortError: Cannot calculate distance")
        return None


    # Sort Functions
    def SORT_DISTANCE_POS(self, rects, pos):
        return sorted(rects, key=lambda r: self._get_distance(pos, self._get_center(r)))


    def SORT_DISTANCE_PRE(self, rects, pos, weights=None):
        if not self.stack:
            return self.SORT_DISTANCE_POS(rects, pos)

        if weights is None:
            weights = [1] * len(self.stack)
        min_len = min(len(self.stack), len(weights))
        weights = weights[:min_len]
        stack = list(reversed(list(self.stack)[-min_len:]))

        return sorted(rects, key=lambda r: np.average([self._get_distance(self._get_center(s), self._get_center(r)) for s in stack], weights=weights))


    # Processing Functions
    def AVERAGE_PRE(self, rect, weights=None):
        rects = list(self.stack) + [rect]

        if weights is None:
            weights = [1] * len(rects)
        min_len = min(len(rects), len(weights))
        weights = weights[:min_len]
        rects = list(reversed(rects[-min_len:]))

        centers = [self._get_center(r) for r in rects]

        avg_cx = np.average([c[0] for c in centers], weights=weights)
        avg_cy = np.average([c[1] for c in centers], weights=weights)
        avg_w = np.average([r[2] for r in rects], weights=weights)
        avg_h = np.average([r[3] for r in rects], weights=weights)

        avg_x = avg_cx - avg_w / 2
        avg_y = avg_cy - avg_h / 2

        return tuple(map(int, (avg_x, avg_y, avg_w, avg_h)))


    # Select Functions
    def SEL_CENTER(self, rect):
        return tuple(map(int, self._get_center(rect)))


    def SEL_DISTANCE_POS(self, rect, pos):
        x, y, w, h = rect

        x_valid = pos[0] is not None
        y_valid = pos[1] is not None
        if x_valid and y_valid:
            return max([(x, y), (x+w, y), (x, y+h), (x+w, y+h)], key=lambda p: self._get_distance(pos, p))
        if x_valid:
            return (max([x, x+w], key=lambda p: abs(pos[0]-p)), y+h//2)
        if y_valid:
            return (x+w//2, max([y, y+h], key=lambda p: abs(pos[1]-p)))
        logging.error("SortError: Cannot calculate distance")
        return None


    def SEL_DISTANCE_POS_POWER(self, rect, pos, max_size):
        _, _, w, h = rect
        cx, cy = self.SEL_CENTER(rect)
        dx, dy = self.SEL_DISTANCE_POS(rect, pos)

        x_valid = pos[0] is not None and max_size[0] is not None
        y_valid = pos[1] is not None and max_size[1] is not None

        if x_valid and y_valid:
            return tuple(map(int, (np.average([cx, dx], weights=[1-(w/max_size[0])**2, (w/max_size[0])**2]), np.average([cy, dy], weights=[-(w/max_size[1])**2, (w/max_size[1])**2]))))
        if x_valid:
            return tuple(map(int, (np.average([cx, dx], weights=[1-(w/max_size[0])**2, (w/max_size[0])**2]), cy)))
        if y_valid:
            return tuple(map(int, (cx, np.average([cy, dy], weights=[1-(w/max_size[1])**2, (w/max_size[1])**2]))))
        return None