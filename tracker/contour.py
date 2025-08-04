import numpy as np
import cv2

from .calc_utils import *



# contour extract functions
def _extract_contours(mask, min_area=0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def extract_convexHulls(mask, min_area=0):
    contours = _extract_contours(mask, min_area)
    hulls = map(lambda c: cv2.convexHull(c, clockwise=True), contours)
    return list(hulls)


def extract_approxPolyDP(mask, epsilon=0.01, min_area=0):
    contours = _extract_contours(mask, min_area)
    polyDPs = map(lambda c: cv2.approxPolyDP(c, epsilon*cv2.arcLength(c, True), True), contours)
    return list(polyDPs)



# contour sort function
def sort_contour_area(contours):
    return sorted(contours, key=cv2.contourArea, reverse=True)



# contour select function
def sel_contour_distance_point(contour, pos, num=1):
    pos = sorted(contour[:, 0, :], key=lambda p: distance(p, pos), reverse=True)
    return tuple(map(int, np.average(pos[:num], axis=0)))