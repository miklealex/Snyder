from interval import interval
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from interval import imath
import math
from collections import deque
import sys
from matplotlib import colors
from collections import OrderedDict 

class MultiPlotViewer:
    def __init__(self, fig, ax_list):
        self.fig = fig
        self.ax_list = ax_list
        self.index = 0

        # Display the first plot initially
        self.update()

    def update(self):
        # Hide all plots
        for ax in self.ax_list:
            ax.set_visible(False)
        
        # Display the current plot
        self.ax_list[self.index].set_visible(True)
        plt.draw()

    def next_plot(self, event):
        if event.key == "right":
            self.index = (self.index + 1) % len(self.ax_list)
            self.update()
        elif event.key == "left":
            self.index = (self.index - 1) % len(self.ax_list)
            self.update()

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
def add_lines(interval_x, interval_y, boundary_eps, lines, global_par_coordinate):
    intersections = []
    intersected_sides = []

    # 1. Calculate all intersection points with the boundary.
    # Check top and bottom horizontal boundaries
    for x_boundary in [interval_x[0].inf, interval_x[0].sup]:
        f = lambda y: implicit_function(x_boundary, y)
        intersection_points = []
        if global_par_coordinate == "xy":
            point = binary_search_intersection(f, interval_y[0].inf, interval_y[0].sup, boundary_eps)
            if point is not None:
                intersection_points.append(point)
        else:
            point = binary_search_intersection(f, interval_y[0].inf, midpoint(interval_y), boundary_eps)
            if point is not None:
                intersection_points.append(point)
            point = binary_search_intersection(f, midpoint(interval_y), interval_y[0].sup, boundary_eps)
            if point is not None:
                intersection_points.append(point)
        if len(intersection_points) > 0:
            side = 'right' if x_boundary == interval_x[0].sup else 'left'
            intersected_sides.append(side)
            for pt in intersection_points:
                intersections.append((x_boundary, pt))

    # Check left and right vertical boundaries
    for y_boundary in [interval_y[0].inf, interval_y[0].sup]:
        f = lambda x: implicit_function(x, y_boundary)
        intersection_points = []
        if global_par_coordinate == "xy":
            point = binary_search_intersection(f, interval_x[0].inf, interval_x[0].sup, boundary_eps)
            if point is not None:
                intersection_points.append(point)
        else:
            point = binary_search_intersection(f, interval_x[0].inf, midpoint(interval_x), boundary_eps)
            if point is not None:
                intersection_points.append(point)
            point = binary_search_intersection(f, midpoint(interval_x), interval_x[0].sup, boundary_eps)
            if point is not None:
                intersection_points.append(point)
        if len(intersection_points) > 0:
            side = 'top' if y_boundary == interval_y[0].sup else 'bottom'
            intersected_sides.append(side)
            for pt in intersection_points:
                intersections.append((pt, y_boundary))
    
    intersections = list(OrderedDict.fromkeys(intersections)) 
    # 2. Sort the intersections
    sorted_intersections = sorted(intersections, key=lambda point: point[0 if global_par_coordinate == 'x' else 1])

    # 3. Calculate the midpoint between each pair of sorted intersection points.
    for i in range(len(sorted_intersections) - 1):
        x1, y1 = sorted_intersections[i]
        x2, y2 = sorted_intersections[i + 1]

        if global_par_coordinate == 'x':
            mid_x = (x1 + x2) / 2
            f = lambda y: implicit_function(mid_x, y)
            intersection = binary_search_intersection(f, interval_y[0].inf, interval_y[0].sup, boundary_eps)
        else:  # parameter == 'y'
            mid_y = (y1 + y2) / 2
            f = lambda x: implicit_function(x, mid_y)
            intersection = binary_search_intersection(f, interval_x[0].inf, interval_x[0].sup, boundary_eps)

        # 4. Check if the implicit function intersects with an imaginary line at this midpoint.
        if intersection:
            # 5. If it does, connect the two intersection points with a line.
            lines.append((sorted_intersections[i], sorted_intersections[i + 1]))

    return intersected_sides

def get_global_parameterization_coordinate(current_interval_x, current_interval_y):
    global_x = is_globally_parameterizable_x(current_interval_x, current_interval_y)
    global_y = is_globally_parameterizable_y(current_interval_x, current_interval_y)

    if global_x or global_y:
        if not global_y:
            return "x"
        elif global_x:
            return "xy"
        return "y"
    return ""

def bfs_subdivide(interval_x, interval_y, lines, boundary_eps=1e-5):
    rectangles = []
    points = []
    depth_rectangles = {} # Store rectangles per depth

    # Initial interval to start BFS
    initial_data = {
        "interval_x": interval_x,
        "interval_y": interval_y,
        "depth": 1
    }
    queue = deque([initial_data])
    interval_sides = {}

    final_intervals = []

    while queue:
        current = queue.popleft()
        cur_interval_x, cur_interval_y, cur_depth = current["interval_x"], current["interval_y"], current["depth"]

        if cur_depth not in depth_rectangles:
            depth_rectangles[cur_depth] = []
            depth_rectangles[cur_depth].extend(final_intervals)
        depth_rectangles[cur_depth].append((cur_interval_x, cur_interval_y))

        global_x = is_globally_parameterizable_x(cur_interval_x, cur_interval_y)
        global_y = is_globally_parameterizable_y(cur_interval_x, cur_interval_y)

        if is_globally_parameterizable(cur_interval_x, cur_interval_y):
            global_par_coordinate = get_global_parameterization_coordinate(cur_interval_x, cur_interval_y)
            intersected_sides = add_lines(cur_interval_x, cur_interval_y, boundary_eps, lines, global_par_coordinate)
            if intersected_sides:
                        interval_sides[(cur_interval_x, cur_interval_y)] = intersected_sides
            rectangles.append((cur_interval_x, cur_interval_y))
            final_intervals.append((cur_interval_x, cur_interval_y))
            continue

        subintervals = []

        x_mid = midpoint(cur_interval_x)
        y_mid = midpoint(cur_interval_y)

        if global_x and not global_y:
            print("global_x and not global_y")
            subintervals = [
                (cur_interval_x, interval([cur_interval_y[0].inf, y_mid])),
                (cur_interval_x, interval([y_mid, cur_interval_y[0].sup]))
            ]
        elif global_y and not global_x:
            print("global_y and not global_x")
            subintervals = [
                (interval([cur_interval_x[0].inf, x_mid]), cur_interval_y),
                (interval([x_mid, cur_interval_x[0].sup]), cur_interval_y)
            ]
        elif not global_x and not global_y:
            #print("global_y and not global_x")
            subintervals = [
                (interval([cur_interval_x[0].inf, x_mid]), interval([cur_interval_y[0].inf, y_mid])),
                (interval([x_mid, cur_interval_x[0].sup]), interval([cur_interval_y[0].inf, y_mid])),
                (interval([cur_interval_x[0].inf, x_mid]), interval([y_mid, cur_interval_y[0].sup])),
                (interval([x_mid, cur_interval_x[0].sup]), interval([y_mid, cur_interval_y[0].sup]))
            ]

        for sub_x, sub_y in subintervals:
            if has_root(sub_x, sub_y):
                if not is_globally_parameterizable(sub_x, sub_y):
                    new_data = {
                        "interval_x": sub_x,
                        "interval_y": sub_y,
                        "depth": cur_depth + 1
                    }
                    queue.append(new_data)
                else:
                    global_par_coordinate = get_global_parameterization_coordinate(sub_x, sub_y)
                    intersected_sides = add_lines(sub_x, sub_y, boundary_eps, lines, global_par_coordinate)
                    final_intervals.append((sub_x, sub_y))
                    if intersected_sides:
                        interval_sides[(sub_x, sub_y)] = intersected_sides
                    rectangles.append((sub_x, sub_y))

    return points, rectangles, interval_sides, depth_rectangles

def plot_by_depth(depth_rectangles):
    fig = plt.figure(figsize=(8, 8))
    ax_list = []

    max_depth = max(depth_rectangles.keys())

    for depth in range(1, max_depth + 1):
        ax = fig.add_subplot(1, 1, 1)
        ax_list.append(ax)
        
        for rect in depth_rectangles[depth]:
            x_ivl, y_ivl = rect
            rect_width = x_ivl[0].sup - x_ivl[0].inf
            rect_height = y_ivl[0].sup - y_ivl[0].inf

            face_rgba = colors.to_rgba('grey', alpha=0.2)
            edge_rgba = colors.to_rgba('black', alpha=1.0)

            ax.add_patch(patches.Rectangle((x_ivl[0].inf, y_ivl[0].inf), rect_width, rect_height, fill=True, facecolor=face_rgba, edgecolor=edge_rgba))
        
        ax.set_title(f"Depth {depth}")
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_visible(False)

    viewer = MultiPlotViewer(fig, ax_list)
    fig.canvas.mpl_connect('key_press_event', viewer.next_plot)
    plt.show()

def final_plot(rectangles, interval_sides, lines):
    fig, ax = plt.subplots()

    # Plot the rectangles
    for rect in rectangles:
        x_ivl, y_ivl = rect
        rect_width = x_ivl[0].sup - x_ivl[0].inf
        rect_height = y_ivl[0].sup - y_ivl[0].inf

        face_rgba = colors.to_rgba('grey', alpha=0.2)
        edge_rgba = colors.to_rgba('black', alpha=1.0)
        ax.add_patch(patches.Rectangle((x_ivl[0].inf, y_ivl[0].inf), rect_width, rect_height, fill=True, facecolor=face_rgba, edgecolor=edge_rgba))

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
    
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    plt.show()

# Define the main function
def main():

    show_mode = "final"
    if len(sys.argv) == 2 and sys.argv[1] == "depth":
        show_mode = "depth"

    x_interval = interval[-1.1, 1.1]
    y_interval = interval[-1.1, 1.1]
    
    lines = [] 

    points, rectangles, interval_sides, depth_rectangles = bfs_subdivide(x_interval, y_interval, lines)

    match show_mode:
        case "final":
            final_plot(rectangles, interval_sides, lines)
        case "depth":
            plot_by_depth(depth_rectangles)

    
    

main()
