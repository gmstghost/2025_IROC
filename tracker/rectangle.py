import numpy as np
import cv2
import logging

from .calc_utils import *



# rectangle extract function
def extract_rects(mask, min_area=0, min_w=0, min_h=0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda c: cv2.contourArea(c)>=min_area, contours)

    rects = map(lambda c: cv2.boundingRect(c), contours)
    rects = filter(lambda r: r[2]>=min_w and r[3]>=min_h, rects)
    return list(rects)



# rectangle sort functions
def sort_rect_distance_point(rects, pos):
    '''Returns rectangles sorted by distance to the point'''
    return sorted(rects, key=lambda r: distance(pos, rect_center(r)))


def sort_rect_distance_stack(rects, stack, weights=None):
    '''Returns rectangles sorted by weighted average distance to the rects in the stack.'''
    if not stack:
        return rects

    if weights is not None:
        min_len = min(len(stack), len(weights))
        weights = weights[-min_len:]
        stack = stack[-min_len:]

    return sorted(rects, key=lambda r: np.average([distance(rect_center(s), rect_center(r)) for s in stack], weights=weights))


def sort_rect_size(rects):
    '''Returns rectanles sorted by size(width*height).'''
    return sorted(rects, key=lambda r: r[2]*r[3], reverse=True)



# point select functions
def sel_rect_center(rect):
    '''Returns center point of the rectangle.'''
    return tuple(map(int, rect_center(rect)))


def sel_rect_distance_corner(rect, pos):
    '''Returns corner of the rectangle farthest to the point.'''
    x, y, w, h = rect

    x_valid = pos[0] is not None
    y_valid = pos[1] is not None
    if x_valid and y_valid:
        return max([(x, y), (x+w, y), (x, y+h), (x+w, y+h)], key=lambda p: distance(pos, p))
    if x_valid:
        return (max([x, x+w], key=lambda p: abs(pos[0]-p)), y+h//2)
    if y_valid:
        return (x+w//2, max([y, y+h], key=lambda p: abs(pos[1]-p)))
    logging.error("SortError: Cannot calculate distance")
    return None


def sel_rect_average_point_power(rect, pos, max_size, power=2):
    '''Returns weighted average point between the center point and the farthest corner.'''
    _, _, w, h = rect
    cx, cy = sel_rect_center(rect)
    dx, dy = sel_rect_distance_corner(rect, pos)

    x_valid = pos[0] is not None and max_size[0] is not None
    y_valid = pos[1] is not None and max_size[1] is not None

    if x_valid and y_valid:
        return tuple(map(int, (np.average([cx, dx], weights=[1-(w/max_size[0])**power,(w/max_size[0])**power]), 
                               np.average([cy, dy], weights=[-(w/max_size[1])**power, (w/max_size[1])**power]))))
    if x_valid:
        return tuple(map(int, (np.average([cx, dx], weights=[1-(w/max_size[0])**power, (w/max_size[0])**power]), cy)))
    if y_valid:
        return tuple(map(int, (cx, np.average([cy, dy], weights=[1-(w/max_size[1])**power, (w/max_size[1])**power]))))
    return None