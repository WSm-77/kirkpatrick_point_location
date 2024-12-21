def mat_det_2x2(a, b, c):
    """
    Calculating the determinant of a 2x2 matrix

    :param a: a tuple of coordinates (x, y) of the first point defining our line
    :param b: a tuple of coordinates (x, y) of the second point defining our line
    :param c: a tuple of coordinates (x, y) of the point which position relative to the line we want to find

    :return: the value of the determinant of the matrix
    """

    ax, ay = a
    bx, by = b
    cx, cy = c
    return (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)

def orient(a, b, c, eps = 10 ** -12):
    """
    Determining the position of point c relative to line ab

    :param a: a tuple of coordinates (x, y) of the first point defining our line
    :param b: a tuple of coordinates (x, y) of the second point defining our line
    :param c: a tuple of coordinates (x, y) of the point which position relative to the line we want to find
    :param eps: float value defining range of values <-eps, eps> which are treated as 0

    :return: 0 - the point lies on the line, 1 - the point lies to the left of the line, -1 - the point lies to the right of the line
    """

    det = mat_det_2x2(a, b, c)
    if abs(det) <= eps:
        return 0
    elif det > 0:
        return 1
    else:
        return -1
