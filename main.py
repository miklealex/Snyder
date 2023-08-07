from interval import interval
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from interval import imath
import math
from collections import deque

# Define the implicit function
def implicit_function(x, y):
    #return x**2 + y**2 - interval[1]
    return x**2 + y**2 + imath.cos(2*imath.pi*x) + imath.sin(2*imath.pi*y) + imath.sin(2*imath.pi*x**2)*imath.cos(2*imath.pi*y**2) - interval[1]

# Calculate the midpoint of an interval
def midpoint(ivl):
    return (ivl[0].inf + ivl[0].sup) / 2

def binary_search_intersection(f, a, b, eps_search=1e-5):
    f_a, f_b = f(a), f(b)

    # Safety Measure: Check if signs are the same
    if (f_a[0].inf > 0 and f_b[0].inf > 0) or (f_a[0].sup < 0 and f_b[0].sup < 0):
        return None

    mid = (a + b) / 2

    if abs(b - a) < eps_search:
        return mid

    f_mid = f(mid)

    # If 0 is in the function value at the midpoint, return it
    if 0 in f_mid:
        return mid

    # If a or b contains 0, return them
    if 0 in f_a:
        return a
    if 0 in f_b:
        return b

    # If f(a) and f(mid) have different signs
    if (f_a[0].inf < 0 and f_mid[0].sup > 0) or (f_a[0].sup > 0 and f_mid[0].inf < 0):
        return binary_search_intersection(f, a, mid, eps_search)
    # If f(mid) and f(b) have different signs
    elif (f_mid[0].inf < 0 and f_b[0].sup > 0) or (f_mid[0].sup > 0 and f_b[0].inf < 0):
        return binary_search_intersection(f, mid, b, eps_search)

    # This shouldn't be reached ideally.
    print("Returning NONE - This point shouldn't be ideally reached")
    return None

def jacobian_determinant(x, y):
    part_x = 2*x - 2*math.pi*imath.sin(2*math.pi*x) + 4*math.pi*x*imath.cos(2*math.pi*x**2)*imath.cos(2*math.pi*y**2)
    part_y = 2*y + 2*math.pi*imath.cos(2*math.pi*y) - 4*math.pi*y**2*imath.sin(2*math.pi*x**2)*imath.sin(2*math.pi*y**2)
    return part_x * part_y

# def is_globally_parameterizable(interval_x, interval_y):
#     det_J = jacobian_determinant(interval_x, interval_y)
#     return 0 not in det_J

def partial_derivative_x(x, y):
    return 2*x - 2*math.pi*imath.sin(2*math.pi*x) + 4*math.pi*x*imath.cos(2*math.pi*x**2)*imath.cos(2*math.pi*y**2)

def partial_derivative_y(x, y):
    return 2*y + 2*math.pi*imath.cos(2*math.pi*y) - 4*math.pi*y**2*imath.sin(2*math.pi*x**2)*imath.sin(2*math.pi*y**2)

def is_globally_parameterizable_x(interval_x, interval_y):
    partial_y = partial_derivative_y(interval_x, interval_y)
    return 0 not in partial_y

def is_globally_parameterizable_y(interval_x, interval_y):
    partial_x = partial_derivative_x(interval_x, interval_y)
    return 0 not in partial_x

def is_globally_parameterizable(interval_x, interval_y):
    return is_globally_parameterizable_x(interval_x, interval_y) or is_globally_parameterizable_y(interval_x, interval_y)


# Determine if the function has a root in the interval
def has_root(interval_x, interval_y):
    return 0 in implicit_function(interval_x, interval_y)

#Note! x and y are flipped for some reason when displaying on the plot
def add_lines(interval_x, interval_y, boundary_eps, lines):
    intersections = []
    intersected_sides = []  # This list stores the sides the function crosses

    # Check top and bottom horizontal boundaries
    for x_boundary in [interval_x[0].inf, interval_x[0].sup]:
        f = lambda y: implicit_function(x_boundary, y)
        intersection = binary_search_intersection(f, interval_y[0].inf, interval_y[0].sup, boundary_eps)
        if intersection is not None:
            # Determine which boundary (top or bottom) has been intersected
            #Changing to right and left because coordinates are flipped
            side = 'right' if x_boundary == interval_x[0].sup else 'left'
            intersected_sides.append(side)
            intersections.append((x_boundary, intersection))
            
    # We continue only if two intersections were found in the previous loop
    if len(intersections) == 2:
        lines.append((intersections[0], intersections[1]))
        return intersected_sides
    
    # Check left and right vertical boundaries
    for y_boundary in [interval_y[0].inf, interval_y[0].sup]:
        f = lambda x: implicit_function(x, y_boundary)
        intersection = binary_search_intersection(f, interval_x[0].inf, interval_x[0].sup, boundary_eps)
        if intersection is not None:
            # Determine which boundary (left or right) has been intersected
            #Changing to top and bottom because coordinates are flipped
            side = 'top' if y_boundary == interval_y[0].sup else 'bottom'
            intersected_sides.append(side)
            intersections.append((intersection, y_boundary))

    if len(intersections) == 2:
        lines.append((intersections[0], intersections[1]))
        return intersected_sides

    #print(intersections)
    return []

def bfs_subdivide(interval_x, interval_y, lines, max_depth=8, eps=0.01, boundary_eps=1e-5):
    rectangles = []
    points = []

    # Initial interval to start BFS
    initial_data = {
        "interval_x": interval_x,
        "interval_y": interval_y,
        "depth": 1
    }
    queue = deque([initial_data])
    interval_sides = {}

    while queue:
        current = queue.popleft()
        cur_interval_x, cur_interval_y, cur_depth = current["interval_x"], current["interval_y"], current["depth"]
        #print("Interval [", cur_interval_x, cur_interval_y)

        if is_globally_parameterizable(cur_interval_x, cur_interval_y):
            intersected_sides = add_lines(cur_interval_x, cur_interval_y, boundary_eps, lines)
            if intersected_sides:
                        interval_sides[(cur_interval_x, cur_interval_y)] = intersected_sides
            rectangles.append((cur_interval_x, cur_interval_y))
            continue

        subintervals = []
        global_x = is_globally_parameterizable_x(cur_interval_x, cur_interval_y)
        global_y = is_globally_parameterizable_y(cur_interval_x, cur_interval_y)
        #print("global_x = ", global_x)
        #print("global_y = ", global_y)
        x_mid = midpoint(cur_interval_x)
        y_mid = midpoint(cur_interval_y)

        if global_x and not global_y:
            subintervals = [
                (cur_interval_x, interval([cur_interval_y[0].inf, y_mid])),
                (cur_interval_x, interval([y_mid, cur_interval_y[0].sup]))
            ]
        elif global_y and not global_x:
            subintervals = [
                (interval([cur_interval_x[0].inf, x_mid]), cur_interval_y),
                (interval([x_mid, cur_interval_x[0].sup]), cur_interval_y)
            ]
        elif not global_x and not global_y:
            subintervals = [
                (interval([cur_interval_x[0].inf, x_mid]), interval([cur_interval_y[0].inf, y_mid])),
                (interval([x_mid, cur_interval_x[0].sup]), interval([cur_interval_y[0].inf, y_mid])),
                (interval([cur_interval_x[0].inf, x_mid]), interval([y_mid, cur_interval_y[0].sup])),
                (interval([x_mid, cur_interval_x[0].sup]), interval([y_mid, cur_interval_y[0].sup]))
            ]

        for sub_x, sub_y in subintervals:
            if has_root(sub_x, sub_y):
                if not is_globally_parameterizable(sub_x, sub_y):
                    #if cur_depth < max_depth:
                    new_data = {
                        "interval_x": sub_x,
                        "interval_y": sub_y,
                        "depth": cur_depth + 1
                    }
                    queue.append(new_data)
                    # else:
                    #     add_lines(sub_x, sub_y, boundary_eps, lines)
                    #     rectangles.append((sub_x, sub_y))
                    #     print("MAX DEPTH REACHED")
                else:
                    intersected_sides = add_lines(sub_x, sub_y, boundary_eps, lines)
                    if intersected_sides:
                        interval_sides[(sub_x, sub_y)] = intersected_sides
                    rectangles.append((sub_x, sub_y))

    return points, rectangles, interval_sides



# Define the main function
def main():
    x_interval = interval[-1.1, 1.1]
    y_interval = interval[-1.1, 1.1]
    
    lines = [] 

    points, rectangles, interval_sides = bfs_subdivide(x_interval, y_interval, lines)
    
    fig, ax = plt.subplots()
    #print(lines)

    # Plot the rectangles
    for rect in rectangles:
        x_ivl, y_ivl = rect
        rect_width = x_ivl[0].sup - x_ivl[0].inf
        rect_height = y_ivl[0].sup - y_ivl[0].inf
        ax.add_patch(patches.Rectangle((x_ivl[0].inf, y_ivl[0].inf), rect_width, rect_height, fill=False))

        sides_to_highlight = interval_sides.get(rect, [])
        for side in sides_to_highlight:
            if side == 'top':
                plt.plot([x_ivl[0].inf, x_ivl[0].sup], [y_ivl[0].sup, y_ivl[0].sup], color='r')
            elif side == 'bottom':
                plt.plot([x_ivl[0].inf, x_ivl[0].sup], [y_ivl[0].inf, y_ivl[0].inf], color='r')
            elif side == 'left':
                plt.plot([x_ivl[0].inf, x_ivl[0].inf], [y_ivl[0].inf, y_ivl[0].sup], color='r')
            elif side == 'right':
                plt.plot([x_ivl[0].sup, x_ivl[0].sup], [y_ivl[0].inf, y_ivl[0].sup], color='r')

    for line in lines:
        (x1, y1), (x2, y2) = line
        plt.plot([x1, x2], [y1, y2], color='b')
    
    plt.show()

main()
