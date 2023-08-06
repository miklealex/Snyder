from interval import interval
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from interval import imath

# Define the implicit function
def implicit_function(x, y):
    #return x**2 + y**2 - interval[1]
    return x**2 + y**2 + imath.cos(2*imath.pi*x) + imath.sin(2*imath.pi*y) + imath.sin(2*imath.pi*x**2)*imath.cos(2*imath.pi*y**2) - interval[1]

# Calculate the midpoint of an interval
def midpoint(ivl):
    return (ivl[0].inf + ivl[0].sup) / 2

def binary_search_intersection(f, a, b, eps_search=1e-7):
    f_a, f_b = f(a), f(b)

    # Safety Measure: Check if signs are the same
    if (f_a[0].inf >= 0 and f_b[0].inf >= 0) or (f_a[0].sup <= 0 and f_b[0].sup <= 0):
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


def is_globally_parameterizable_x(interval_x, interval_y):
    fx1 = implicit_function(interval_x[0].inf, midpoint(interval_y))
    fx2 = implicit_function(interval_x[0].sup, midpoint(interval_y))
    return fx1 * fx2 <= interval[0]

def is_globally_parameterizable_y(interval_x, interval_y):
    fy1 = implicit_function(midpoint(interval_x), interval_y[0].inf)
    fy2 = implicit_function(midpoint(interval_x), interval_y[0].sup)
    return fy1 * fy2 <= interval[0]

# Check if the interval is globally parameterizable
def is_globally_parameterizable(interval_x, interval_y):
    # Check boundaries
    
    return is_globally_parameterizable_x(interval_x, interval_y) or is_globally_parameterizable_y(interval_x, interval_y)

# Determine if the function has a root in the interval
def has_root(interval_x, interval_y):
    return 0 in implicit_function(interval_x, interval_y)

def add_lines(interval_x, interval_y, boundary_eps, lines):
    intersections = []

    # Check top and bottom horizontal boundaries
    for x_boundary in [interval_x[0].inf, interval_x[0].sup]:
        f = lambda y: implicit_function(x_boundary, y)
        intersection = binary_search_intersection(f, interval_y[0].inf, interval_y[0].sup, boundary_eps)
        if intersection is not None:
            #print("intersection y found")
            intersections.append((x_boundary, intersection))
    if len(intersections) == 2:
        lines.append((intersections[0], intersections[1]))
        return True
    
    #intersections.clear()
    # Check left and right vertical boundaries
    for y_boundary in [interval_y[0].inf, interval_y[0].sup]:
        f = lambda x: implicit_function(x, y_boundary)
        intersection = binary_search_intersection(f, interval_x[0].inf, interval_x[0].sup, boundary_eps)
        if intersection is not None:
            #print("intersection x found")
            intersections.append((intersection, y_boundary))

    if len(intersections) == 2:
        lines.append((intersections[0], intersections[1]))
        return True
    
    return False

# Recursive subdivision based on global parameterizability
def subdivide(interval_x, interval_y, lines, depth = 1, max_depth = 6, eps=0.01, boundary_eps=1e-5):
    rectangles = [(interval_x, interval_y)]  # Store current interval as a rectangle
    points = []

    if abs(interval_x[0].sup - interval_x[0].inf) < eps and abs(interval_y[0].sup - interval_y[0].inf) < eps:
        if has_root(interval_x, interval_y):
            points.append((midpoint(interval_x), midpoint(interval_y)))
        add_lines(interval_x, interval_y, boundary_eps, lines)
        return points, rectangles

    x_mid = midpoint(interval_x)
    y_mid = midpoint(interval_y)

    top_left = (interval([interval_x[0].inf, x_mid]), interval([interval_y[0].inf, y_mid]))
    top_right = (interval([x_mid, interval_x[0].sup]), interval([interval_y[0].inf, y_mid]))
    bottom_left = (interval([interval_x[0].inf, x_mid]), interval([y_mid, interval_y[0].sup]))
    bottom_right = (interval([x_mid, interval_x[0].sup]), interval([y_mid, interval_y[0].sup]))

    for subdiv in [top_left, top_right, bottom_left, bottom_right]:
        if is_globally_parameterizable(subdiv[0], subdiv[1]) or has_root(subdiv[0], subdiv[1]):
            if depth >= max_depth:
                if add_lines(interval_x, interval_y, boundary_eps, lines) == True:
                    return [], rectangles
            new_points, new_rectangles = subdivide(subdiv[0], subdiv[1], lines, depth + 1, max_depth, eps, boundary_eps)
            points.extend(new_points)
            rectangles.extend(new_rectangles)
    return points, rectangles

# Define the main function
def main():
    x_interval = interval[-1.1, 1.1]
    y_interval = interval[-1.1, 1.1]
    
    lines = [] 

    points, rectangles = subdivide(x_interval, y_interval, lines)
    
    fig, ax = plt.subplots()
    #print(lines)

    # Plot the rectangles
    for rect in rectangles:
        x_ivl, y_ivl = rect
        rect_width = x_ivl[0].sup - x_ivl[0].inf
        rect_height = y_ivl[0].sup - y_ivl[0].inf
        ax.add_patch(patches.Rectangle((x_ivl[0].inf, y_ivl[0].inf), rect_width, rect_height, fill=False))

    for line in lines:
        (x1, y1), (x2, y2) = line
        plt.plot([x1, x2], [y1, y2], color='b')
    
    # Plot the points
    #if points:
    #    x_values, y_values = zip(*points)
    #    plt.scatter(x_values, y_values, color='r')  # Color set to red for clarity against rectangles
    #else:
    #    print("No points found!")

    #plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

main()
