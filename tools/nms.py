import numpy as np
from shapely.geometry import Polygon


def poly_nms(dets, thresh):
    """

    :param dets:
    :param thresh:
    :return:
    """
    x1, y1 = dets[:, 0], dets[:, 1]
    x2, y2 = dets[:, 2], dets[:, 3]
    x3, y3 = dets[:, 4], dets[:, 5]
    x4, y4 = dets[:, 6], dets[:, 7]
    scores = dets[:, 8]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i_max = order[0]
        keep.append(i_max)
        over = []
        for i in xrange(1, len(order)):
            max_scored_poly = Polygon([(x1[i_max], y1[i_max]), (x4[i_max], y4[i_max]),
                                       (x3[i_max], y3[i_max]), (x2[i_max], y2[i_max])])
            other_poly = Polygon([(x1[i], y1[i]), (x4[i], y4[i]),
                                  (x3[i], y3[i]), (x2[i], y2[i])])
            if max_scored_poly.intersects(other_poly):
                inter_area = max_scored_poly.intersection(other_poly)
                over.append(inter_area.area / (max_scored_poly.area + other_poly.area - inter_area.area))
            else:
                over.append(0)

        # inds = np.where(over <= thresh)[0]
        # np.where handle numpy array, can not handle list
        inds = np.where(np.array(over) <= thresh)[0]
        order = order[inds + 1]
    return keep


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # thresh:0.3,0.5....
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    print areas
    order = scores.argsort()[::-1]
    print 'order is {0}'.format(order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        print 'xx1'
        print xx1
        print yy1
        print xx2
        print yy2
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        print 'inter is {0}'.format(inter)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        print 'over is {0}'.format(ovr)
        inds = np.where(ovr <= thresh)[0]
        print 'indds is {0}'.format(inds)
        order = order[inds + 1]
        print 'order is {0}'.format(order)
    print 'keep is {0}'.format(keep)
    return keep


def non_max_suppression_slow(boxes, overlap_thresh):
    """

    :param boxes:
    :param overlap_thresh:
    :return:
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    print boxes
    print area
    idxs = np.argsort(y2)
    print idxs
    print 'over test.....'

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlap_thresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]


if __name__ == '__main__':

    print 'hello'
