import numpy as np
from scipy.spatial import ConvexHull

def intersect_lines(L1, L2, eps=1e-12):
    # L: (a,b,c) representing a*x + b*y = c
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < eps:
        return None
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return np.array([x, y], float)

def feasible(pt, halfspaces, tol=1e-9):
    # halfspaces: list of (a,b,c) meaning a*x + b*y <= c
    x, y = float(pt[0]), float(pt[1])
    for a, b, c in halfspaces:
        if a * x + b * y > c + tol:
            return False
    return True

def polygon_to_halfspaces(poly, tol=1e-12):
    """
    Convert a convex polygon (given in counterclockwise order) into halfspaces
    of the form a*x + b*y <= c describing its interior.
    """
    poly = np.asarray(poly, float)
    n = poly.shape[0]
    halfspaces = []

    # Ensure CCW orientation
    area2 = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    if area2 < 0:
        poly = poly[::-1]

    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        edge = p2 - p1

        # For CCW polygon, inward normal is [-edge_y, edge_x]
        a = -edge[1]
        b = edge[0]
        c = a * p1[0] + b * p1[1]

        halfspaces.append((float(a), float(b), float(c)))

    return halfspaces

def halfspace_polygon_2d(halfspaces, box, tol=1e-9):
    """
    Compute polygon vertices for intersection of halfspaces + bounding box.

    halfspaces: [(a,b,c), ...] meaning a*x + b*y <= c
    box: (x_min,x_max,y_min,y_max)

    Returns ordered vertices (N,2) or None.
    """
    x_min, x_max, y_min, y_max = box

    # Add bounding box as halfspaces
    hs = list(halfspaces)
    hs += [
        ( 1.0,  0.0, x_max),   # x <= x_max
        (-1.0,  0.0, -x_min),  # -x <= -x_min  => x >= x_min
        ( 0.0,  1.0, y_max),   # y <= y_max
        ( 0.0, -1.0, -y_min),  # y >= y_min
    ]

    # Candidate vertices are intersections of boundary lines (equalities)
    candidates = []

    # Convert each inequality to its boundary line for intersection tests
    lines = [(a, b, c) for (a, b, c) in hs]

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            p = intersect_lines(lines[i], lines[j])
            if p is None:
                continue
            if feasible(p, hs, tol=tol):
                candidates.append(p)

    # Also include rectangle corners if feasible
    corners = [
        np.array([x_min, y_min]),
        np.array([x_min, y_max]),
        np.array([x_max, y_min]),
        np.array([x_max, y_max]),
    ]
    for c in corners:
        if feasible(c, hs, tol=tol):
            candidates.append(c.astype(float))

    if len(candidates) < 3:
        return None

    # Deduplicate points
    P = np.vstack(candidates)
    # round-based dedupe
    P = np.unique(np.round(P, 10), axis=0)
    if P.shape[0] < 3:
        return None

    # Order by convex hull
    hull = ConvexHull(P)
    return P[hull.vertices]

def halfspace_intersection_2d(halfspaces, tol=1e-9):
    """
    Compute polygon vertices for the intersection of arbitrary 2D halfspaces.

    halfspaces: [(a,b,c), ...] meaning a*x + b*y <= c

    Returns ordered vertices (N,2) or None.
    """
    candidates = []
    lines = [(a, b, c) for (a, b, c) in halfspaces]

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            p = intersect_lines(lines[i], lines[j])
            if p is None:
                continue
            if feasible(p, halfspaces, tol=tol):
                candidates.append(p)

    if len(candidates) < 3:
        return None

    P = np.vstack(candidates)
    P = np.unique(np.round(P, 10), axis=0)

    if P.shape[0] < 3:
        return None

    hull = ConvexHull(P)
    return P[hull.vertices]
