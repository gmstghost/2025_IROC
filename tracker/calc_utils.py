import logging


# 직사각형 중심 찾기
def rect_center(rect):
    x, y, w, h = rect
    return (x + w/2, y + h/2)


def distance(pos1, pos2):
    # 
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