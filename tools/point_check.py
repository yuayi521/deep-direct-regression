def point_in_polygon(x, y, verts):
    """
    - http://blog.leanote.com/post/iwantpython@163.com/d4d62c5dc860
    - PNPoly algorithm
    - xyverts  [(x1, y1), (x2, y2), (x3, y3), ...]
    """
    try:
        x, y = float(x), float(y)
    except:
        return False
    vertx = [xyvert[0] for xyvert in verts]
    verty = [xyvert[1] for xyvert in verts]

    if not verts or not min(vertx) <= x <= max(vertx) or not min(verty) <= y <= max(verty):
        return False

    nvert = len(verts)
    is_in = False
    for i in range(nvert):
        j = nvert - 1 if i == 0 else i - 1
        if ((verty[i] > y) != (verty[j] > y)) and (
                    x < (vertx[j] - vertx[i]) * (y - verty[i]) / (verty[j] - verty[i]) + vertx[i]):
            is_in = not is_in

    return is_in
