import numpy as np

import camera_functions

from fill_triangles import gourauds

def render_object(p3d, faces, vcolors, H, W, rows, columns, f, cv, cK, cup):
    # convert points to 2d camera space
    p2d, depth = camera_functions.camera_looking_at(f, cv, cK, cup, p3d)
    verts2d = camera_functions.rasterize(p2d, rows, columns, H, W)

    # grab useful quantities
    k = len(faces) 

    # construct initial image
    img = np.ones((rows, columns, 3))

    # find all triangles, with the proper order according to depth
    # constuct cog depths hashmap
    cog_depths = np.zeros((k))
    cog_depths = {}
    for i in range(k):
        # first coordinate
        idx_a = faces[i][0]
        idx_b = faces[i][1]
        idx_c = faces[i][2]

        # find points
        point_a_depth = depth[idx_a]
        point_b_depth = depth[idx_b]
        point_c_depth = depth[idx_c]

        # calculate cog coordinate in z axis (depth)
        cog_depth = (point_a_depth + point_b_depth + point_c_depth) / 3.0

        # store depth (z coordinate) in hashmap
        cog_depths[str(cog_depth)] = i

    # color all triangles
    # obtain all depths (keys of dictionary, strings)
    depths = list(cog_depths.keys())
    # sort in reverse, since we want further triangles
    # (higher depth) to be colored firsts
    # sort by numerical value
    depths.sort(key=lambda x: float(x), reverse=True)
    # iterate and draw triangle
    for depth in depths:
        # get index
        faces_idx = cog_depths[depth]
        # get point idices
        point_idx = faces[faces_idx]

        # get vertices
        triangle = np.array(verts2d[point_idx])
        triangle = triangle.astype(np.int64)
        # get vertices' colors
        colors = np.array(vcolors[point_idx])

        # return img
        # paint triangle over canvas
        # only if valid coordinates
        plot_triangle = True
        for point in triangle:
            if (point[0] < 0 or point[0] > rows - 1) or \
                (point[1] < 0 or point[1] > columns - 1):
                plot_triangle = False

        if plot_triangle:
            img = gourauds(img, triangle, colors)

    # return image
    return img
