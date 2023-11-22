from interval import interval
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from interval import imath
import math
from collections import deque
import sys
from matplotlib import colors
from collections import OrderedDict 
import time

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

def implicit_function(x, y):
    #return x**2 + y**2 - 0.5
    #return x**2 + y**2 + math.cos(2*math.pi*x) + math.sin(2*math.pi*y) + math.sin(2*math.pi*x**2)*math.cos(2*math.pi*y**2) - 1
    return x**3 - y**3 + math.cos(math.pi * x) + math.sin(math.pi * y) + math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y) - 1
    #return math.tan(x**2 + y**2) - 1
    #return math.exp(-x**2 - y**2) * math.sin(x) * math.cos(y) + math.log(1 + x**2 + y**2)
    #return x**2 - y
    #return math.exp(-(x**2 + y**2) / 10) * (math.cos(3*x) * math.cos(3*y) + math.sin(5*x) * math.sin(5*y))

# Define the implicit function
def implicit_function_interval(x, y):
    #return x**2 + y**2 - interval[0.5]
    return x**3 - y**3 + imath.cos(imath.pi * x) + imath.sin(imath.pi * y) + imath.sin(2 * imath.pi * x) * imath.cos(2 * imath.pi * y) - 1
    #return x**2 + y**2 + imath.cos(2*imath.pi*x) + imath.sin(2*imath.pi*y) + imath.sin(2*imath.pi*x**2)*imath.cos(2*imath.pi*y**2) - interval[1]
    #return imath.tan(x**2 + y**2) - 1
    #return imath.exp(-x**2 - y**2) * imath.sin(x) * imath.cos(y) + imath.log(1 + x**2 + y**2)
    #return x**2 - y
    #return imath.exp(-(x**2 + y**2) / 10) * (imath.cos(3*x) * imath.cos(3*y) + imath.sin(5*x) * imath.sin(5*y))

def partial_derivative_x(x, y):
    #return 2*x
    return 3 * x**2 - math.pi * math.sin(math.pi * x) + 2 * math.pi * math.cos(2 * math.pi * x) * math.cos(2 * math.pi * y)
    #return 2*x - 2*math.pi*math.sin(2*math.pi*x) + 4*math.pi*x*math.cos(2*math.pi*x**2)*math.cos(2*math.pi*y**2)
    #return 2*x / math.cos(x**2 + y**2)**2
    #return -2*x*math.exp(-x**2 - y**2) * math.sin(x) * math.cos(y) + math.exp(-x**2 - y**2) * math.cos(x) * math.cos(y) + 2*x / (1 + x**2 + y**2)
    #return 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
    #return 2*x
    # common_factor = math.exp(-(x**2 + y**2) / 10)
    # return -x/5 * common_factor * (math.cos(3*x) * math.cos(3*y) + math.sin(5*x) * math.sin(5*y)) + \
    #        common_factor * (-3*math.sin(3*x) * math.cos(3*y) + 5*math.cos(5*x) * math.sin(5*y))

def partial_derivative_y(x, y):
    #return 2*y
    return -3 * y**2 + math.pi * math.cos(math.pi * y) - 2 * math.pi * math.sin(2 * math.pi * y) * math.sin(2 * math.pi * x)
    #return 2*y + 2*math.pi*math.cos(2*math.pi*y) - 4*math.pi*y*math.sin(2*math.pi*y**2)*math.sin(2*math.pi*x**2)
    #return 2*y / math.cos(x**2 + y**2)**2
    #return -2*y*math.exp(-x**2 - y**2) * math.sin(x) * math.cos(y) - math.exp(-x**2 - y**2) * math.sin(x) * math.sin(y) + 2*y / (1 + x**2 + y**2)
    #return 1
    # common_factor = math.exp(-(x**2 + y**2) / 10)
    # return -y/5 * common_factor * (math.cos(3*x) * math.cos(3*y) + math.sin(5*x) * math.sin(5*y)) + \
    #        common_factor

def partial_derivative_x_interval(x, y):
    #return 2*x
    return 3 * x**2 - imath.pi * imath.sin(imath.pi * x) + 2 * imath.pi * imath.cos(2 * imath.pi * x) * imath.cos(2 * imath.pi * y)
    #return 2*x - 2*math.pi*imath.sin(2*math.pi*x) + 4*math.pi*x*imath.cos(2*math.pi*x**2)*imath.cos(2*math.pi*y**2)
    #return 2*x / imath.cos(x**2 + y**2)**2
    #return -2*x*imath.exp(-x**2 - y**2) * imath.sin(x) * imath.cos(y) + imath.exp(-x**2 - y**2) * imath.cos(x) * imath.cos(y) + 2*x / (1 + x**2 + y**2)
    #return 2*x
    # common_factor = imath.exp(-(x**2 + y**2) / 10)
    # return -x/5 * common_factor * (imath.cos(3*x) * imath.cos(3*y) + imath.sin(5*x) * imath.sin(5*y)) + \
    #        common_factor * (-3*imath.sin(3*x) * imath.cos(3*y) + 5*imath.cos(5*x) * imath.sin(5*y))

def partial_derivative_y_interval(x, y):
    #return 2*y
    return -3 * y**2 + imath.pi * imath.cos(imath.pi * y) - 2 * imath.pi * imath.sin(2 * imath.pi * y) * imath.sin(2 * imath.pi * x)
    #return 2*y + 2*math.pi*imath.cos(2*math.pi*y) - 4*math.pi*y*imath.sin(2*math.pi*y**2)*imath.sin(2*math.pi*x**2)
    #return 2*y / imath.cos(x**2 + y**2)**2
    #return -2*y*imath.exp(-x**2 - y**2) * imath.sin(x) * imath.cos(y) - imath.exp(-x**2 - y**2) * imath.sin(x) * imath.sin(y) + 2*y / (1 + x**2 + y**2)
    #return interval[1]
    # common_factor = imath.exp(-(x**2 + y**2) / 10)
    # return -y/5 * common_factor * (imath.cos(3*x) * imath.cos(3*y) + imath.sin(5*x) * imath.sin(5*y)) + \
    #        common_factor

# Calculate the midpoint of an interval
def midpoint(ivl):
    if isinstance(ivl, float):
        return ivl
    if not ivl:
        return None
    return (ivl[0].inf + ivl[0].sup) / 2

def width(ivl):
    if not ivl:
        return None
    return math.fabs(ivl[0].sup - ivl[0].inf)

def subinverval(first, second):
    if not first or not second:
        return False
    return first[0].inf >= second[0].inf and first[0].sup <= second[0].sup

iterations = 0

def binary_search_internal(f, a, b, eps_search=1e-2):
    f_a, f_b = f(a), f(b)

    global iterations
    iterations += 1
    # Safety Measure: Check if signs are the same
    if (f_a > 0 and f_b > 0) or (f_a < 0 and f_b < 0):
        return None

    mid = (a + b) / 2

    if abs(b - a) < eps_search:
        return interval([a, b])

    f_mid = f(mid)

    # If 0 is in the function value at the midpoint, return it
    if 0 == f_mid:
        return interval([mid])

    # If a or b contains 0, return them
    if 0 == f_a:
        return interval([a])
    if 0 == f_b:
        return interval([b])

    # If f(a) and f(mid) have different signs
    if (f_a < 0 and f_mid > 0) or (f_a > 0 and f_mid < 0):
        return binary_search_internal(f, a, mid, eps_search)
    # If f(mid) and f(b) have different signs
    elif (f_mid < 0 and f_b > 0) or (f_mid > 0 and f_b < 0):
        return binary_search_internal(f, mid, b, eps_search)

    # This shouldn't be reached ideally.
    print("Returning NONE - This point shouldn't be ideally reached")
    return None

def newton_interval_method(f, i_f, i_df, initial_interval, tolerance=1e-5, max_iterations=100):
    current_interval = initial_interval
    intervals = []

    for index in range(max_iterations):
        m = midpoint(current_interval)
        global iterations
        iterations += 1

        derivative_interval = i_df(current_interval)
        next_interval = m - (f(m) / derivative_interval)
        prev_interval = current_interval
        current_interval = next_interval & current_interval
        
        if current_interval and width(current_interval) < tolerance:
            if not 0 in i_f(current_interval):
                return None
            break
        elif current_interval and len(current_interval) > 1:
            first = newton_interval_method(f, i_f, i_df, interval(current_interval[0]), tolerance, max_iterations)
            second = newton_interval_method(f, i_f, i_df, interval(current_interval[1]), tolerance, max_iterations)
            if first != None:
                intervals.extend(first)
            if second != None:
                intervals.extend(second)
            if len(intervals) == 0:
                return None
            return intervals
        # elif current_interval and width(current_interval) >= width(prev_interval):
        #     first = newton_interval_method(f, i_f, i_df, interval([initial_interval[0].inf, midpoint(initial_interval)]), tolerance, max_iterations)
        #     second = newton_interval_method(f, i_f, i_df, interval([midpoint(initial_interval), initial_interval[0].sup]), tolerance, max_iterations)
        #     if first != None:
        #         intervals.extend(first)
        #     if second != None:
        #         intervals.extend(second)
        #     if len(intervals) == 0:
        #         return None
        #     return intervals
        elif not current_interval:
            return None

    intervals.append(current_interval)
    return intervals

# 1. Як обчислюється похідна (як ми переходимо з нелінійної функції в лінійну) - працює, але треба роздуплитися
# 2. Реалізація похідної (чи інтервал найменшої ширини - якщо переставити місцями доданки чи множники, інтервал міняється)
# 3. Показати на малюнку інтервал, в якому лежить крива (а не середню точну)
# 4. Неінтервальна похідна в методі типу рунге (перший доданок в знаменнику)
# 5. Якщо декілька коренів - методи ньютона і рунге не збігатимуться (на якійсь ітерації отримаємо НЕ менший інтервал. Тут потрібно розбити на два (мб, рекурсивно))
# 6. Порахувати і час виконання і к-сть ітерацій
# 7. До нового року закінчити з 2-вимірним випадком
# 8. Кешувати вже знайдені точки(інтервали перетину) щоб не перераховувати.

def runge_interval_method(f, i_f, df, i_df, initial_interval, tolerance=1e-5, max_iterations=100):
    current_interval = initial_interval
    intervals = []

    for index in range(max_iterations):
        m = midpoint(current_interval)
        global iterations
        iterations += 1

        runge_interval = (1/4)*df(m) + (3/4)*i_df(m + (2/3)*(current_interval - m))
        next_interval = m - (f(m) / runge_interval)

        current_interval = next_interval & current_interval
        
        if current_interval and width(current_interval) < tolerance:
            # if not 0 in i_f(current_interval):
            #     return None
            break
        elif current_interval and len(current_interval) > 1:
            first = runge_interval_method(f, i_f, df, i_df, interval(current_interval[0]), tolerance, max_iterations)
            second = runge_interval_method(f, i_f, df, i_df, interval(current_interval[1]), tolerance, max_iterations)
            if first != None:
                intervals.extend(first)
            if second != None:
                intervals.extend(second)
            if len(intervals) == 0:
                return None
            return intervals
        elif not current_interval:
            return None

    intervals.append(current_interval)
    return intervals

def is_globally_parameterizable_x(interval_x, interval_y):
    partial_x = partial_derivative_x_interval(interval_x, interval_y)
    return 0 not in partial_x

def is_globally_parameterizable_y(interval_x, interval_y):
    partial_y = partial_derivative_y_interval(interval_x, interval_y)
    return 0 not in partial_y

def is_globally_parameterizable(interval_x, interval_y):
    if not has_root(interval_x, interval_y):
        return False, "no root"
    
    is_x = is_globally_parameterizable_x(interval_x, interval_y)
    is_y = is_globally_parameterizable_y(interval_x, interval_y)
    if is_x :
        if is_y:
            return True, "xy"
        return True, "x"
    if is_y:
        return True, "y"
    return False, ""


# Determine if the function has a root in the interval
def has_root(interval_x, interval_y):
    return 0 in implicit_function_interval(interval_x, interval_y)

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
        ax.set_xlim([-2.1, 2.1])
        ax.set_ylim([-2.1, 2.1])
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
    
    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])
    plt.show()

def intersectionMethodInternal(f, i_f, df, i_df, ivl, intersection_method, tolerance = 1e-5):
    points = []

    match intersection_method:
        case "binary":
            left = binary_search_internal(f, ivl[0].inf, midpoint(ivl), tolerance)
            if left != None:
                points.append(left)
            right = binary_search_internal(f, midpoint(ivl), ivl[0].sup, tolerance)
            if right != None:
                points.append(right)
            return points
        case "newton":
            res = newton_interval_method(f, i_f, i_df, interval([ivl[0].inf, ivl[0].sup]), tolerance)
            if res != None:
                points.extend(res)
            # left = newton_interval_method(f, i_f, i_df, interval([ivl[0].inf, midpoint(ivl)]), tolerance)
            # if left != None:
            #     points.extend(left)
            # right = newton_interval_method(f, i_f, i_df, interval([midpoint(ivl), ivl[0].sup]), tolerance)
            # if right != None:
            #     points.extend(right)

            if len(points) > 1:
                unique_points = []
                unique_points.append(points[0])
                n = len(points)
                for i in range(1, len(points)):
                    unique = True
                    for j in range(0, len(unique_points)):
                        res = math.fabs(points[i][0].inf - unique_points[j][0].inf)
                        if round(res, int(-math.log(tolerance, 10))) <= tolerance:
                            unique = False
                            break
                    if unique == True:
                        unique_points.append(points[i])
                return unique_points
                           
            return points
        case "runge":
            res = runge_interval_method(f, i_f, df, i_df, interval([ivl[0].inf, ivl[0].sup]), tolerance)
            if res != None:
                points.extend(res)
            # left = runge_interval_method(f, i_f, df, i_df, interval([ivl[0].inf, midpoint(ivl)]), tolerance)
            # if left != None:
            #     points.append(left)
            # right = runge_interval_method(f, i_f, df, i_df, interval([midpoint(ivl), ivl[0].sup]), tolerance)
            # if right != None:
            #     points.append(right)
            
            if len(points) > 1:
                unique_points = []
                unique_points.append(points[0])
                n = len(points)
                for i in range(1, len(points)):
                    unique = True
                    for j in range(0, len(unique_points)):
                        res = math.fabs(points[i][0].inf - unique_points[j][0].inf)
                        if round(res, int(-math.log(tolerance, 10))) <= tolerance:
                            unique = False
                            break
                    if unique == True:
                        unique_points.append(points[i])
                return unique_points
            
            return points

    return {}

intersections_cache = {"y":{}, "x":{}}
cache_hit_counter = 0

# "y": {
#   y1: [intx1, intx2, ...],
#   ...
# },
# "x" : {
#   x1: [inty1, inty2, ...],
#   ...
# }

def getIntersectionPoints(interval_x, interval_y, intersection_method, tolerance = 1e-5):
    intersections = {}
    global intersections_cache
    global cache_hit_counter

    if interval_x[0].inf > -0.5 and interval_x[0].sup < -0.3 and interval_y[0].inf > 0.3 and interval_y[0].sup < 0.5:
        print("probably this one")

    # Check left and right vertical boundaries
    for x_boundary in [interval_x[0].inf, interval_x[0].sup]:
        coords_to_side = {interval_x[0].inf: "left", interval_x[0].sup: "right"}
        if x_boundary in intersections_cache["x"]:
            values = intersections_cache["x"][x_boundary]
            foundCacheHit = []
            for result in values:
                x, ivl = result
                if ivl[0].inf >= interval_y[0].inf and ivl[0].sup <= interval_y[0].sup:
                    foundCacheHit.append(result)
                    cache_hit_counter += 1
            if len(foundCacheHit) > 0:
                intersections[coords_to_side[x_boundary]] = foundCacheHit
                continue
        f = lambda y: implicit_function(x_boundary, y)
        i_f = lambda y: implicit_function_interval(x_boundary, y)
        df = lambda y: partial_derivative_y(x_boundary, y)
        df_int = lambda y: partial_derivative_y_interval(x_boundary, y)
        points = intersectionMethodInternal(f, i_f, df, df_int, interval_y, intersection_method, tolerance)
        if points != None and len(points) > 0:
            for pt in points:
                if not coords_to_side[x_boundary] in intersections:
                    intersections[coords_to_side[x_boundary]] = []
                intersections[coords_to_side[x_boundary]].append((x_boundary, pt))
            if not x_boundary in intersections_cache["x"]:
                intersections_cache["x"][x_boundary] = []
            intersections_cache["x"][x_boundary].extend(intersections[coords_to_side[x_boundary]])
        

    # Check top and bottom horizontal boundaries
    for y_boundary in [interval_y[0].inf, interval_y[0].sup]:
        coords_to_side = {interval_y[0].inf: "bottom", interval_y[0].sup: "top"}
        if y_boundary in intersections_cache["y"]:
            values = intersections_cache["y"][y_boundary]
            foundCacheHit = []
            for result in values:
                ivl, y = result
                if ivl[0].inf >= interval_x[0].inf and ivl[0].sup <= interval_x[0].sup:
                    foundCacheHit.append(result)
                    cache_hit_counter += 1
            if len(foundCacheHit) > 0:
                intersections[coords_to_side[y_boundary]] = foundCacheHit
                continue
        f = lambda x: implicit_function(x, y_boundary)
        i_f = lambda x: implicit_function_interval(x, y_boundary)
        df = lambda x: partial_derivative_x(x, y_boundary)
        df_int = lambda x: partial_derivative_x_interval(x, y_boundary)
        points = intersectionMethodInternal(f, i_f, df, df_int, interval_x, intersection_method, tolerance)
        if points != None and len(points) > 0:
            for pt in points:
                if not coords_to_side[y_boundary] in intersections:
                    intersections[coords_to_side[y_boundary]] = []
                intersections[coords_to_side[y_boundary]].append((pt, y_boundary))
            if not y_boundary in intersections_cache["y"]:
                intersections_cache["y"][y_boundary] = []
            intersections_cache["y"][y_boundary].extend(intersections[coords_to_side[y_boundary]])
    
    return intersections

def shouldSubdivide(intersections):
    for side in intersections:
        if len(intersections[side]) > 1:
            return True
    return False

def intersectionsSortingInternal(point):
    if isinstance(point[0], float):
        return point[0]
    return midpoint(point[0])

def buildLines(intersections, g_p_option, intersection_method, tolerance = 1e-5):

    lines = []
    points = []
    sides = []
    for key in intersections:
        if len(intersections[key]) > 0:
            points.extend(intersections[key])
            sides.append(key)
    if len(points) < 2:
        return []
    sortedPoints = list(OrderedDict.fromkeys(points)) 

    # 2. Sort the intersections
    sorted_intersections = sorted(sortedPoints, key=lambda point: intersectionsSortingInternal(point))
    # 3. Calculate the midpoint between each pair of sorted intersection points.
    for i in range(len(sorted_intersections) - 1):
        x1, y1 = sorted_intersections[i]
        x2, y2 = sorted_intersections[i + 1]
        middle_intersection_points = []
        s1 = s2 = "unknown"
        if "top" in sides:
            if sorted_intersections[i] in intersections["top"]:
                s1 = "top"
            elif sorted_intersections[i + 1] in intersections["top"]:
                s2 = "top"

        if g_p_option == 'x':
            mid_x = (x1 + x2) / 2
            f = lambda y: implicit_function(midpoint(mid_x), y)
            i_f = lambda y: implicit_function_interval(midpoint(mid_x), y)
            df = lambda y: partial_derivative_y(midpoint(mid_x), y)
            df_int = lambda y: partial_derivative_y_interval(midpoint(mid_x), y)
            middle_intersection_points = intersectionMethodInternal(f, i_f, df, df_int, interval([y1, y2]), intersection_method, tolerance)
        else:  # parameter == 'y'
            mid_y = (y1 + y2) / 2
            f = lambda x: implicit_function(x, midpoint(mid_y))
            i_f = lambda x: implicit_function_interval(x, midpoint(mid_y))
            df = lambda x: partial_derivative_x(x, midpoint(mid_y))
            df_int = lambda x: partial_derivative_x_interval(x, midpoint(mid_y))
            middle_intersection_points = intersectionMethodInternal(f, i_f, df, df_int, interval([x1, x2]), intersection_method, tolerance)
        # 4. Check if the implicit function intersects with an imaginary line at this midpoint.
        if middle_intersection_points != None and len(middle_intersection_points) > 0:
            # 5. If it does, connect the two intersection points with a line.
            if isinstance(x1, interval) and isinstance(y2, interval):
                if s1 == "top":
                    lines.append(((x1[0].inf, y1), (x2, y2[0].inf)))
                    lines.append(((x1[0].sup, y1), (x2, y2[0].sup)))
                else:
                    lines.append(((x1[0].inf, y1), (x2, y2[0].sup)))
                    lines.append(((x1[0].sup, y1), (x2, y2[0].inf)))
            elif isinstance(y1, interval) and isinstance(x2, interval):
                if s2 == "top":
                    lines.append(((x1, y1[0].inf), (x2[0].sup, y2)))
                    lines.append(((x1, y1[0].sup), (x2[0].inf, y2)))
                else:
                    lines.append(((x1, y1[0].inf), (x2[0].inf, y2)))
                    lines.append(((x1, y1[0].sup), (x2[0].sup, y2)))
            elif isinstance(x1, interval) and isinstance(x2, interval):
                lines.append(((x1[0].inf, y1), (x2[0].inf, y2)))
                lines.append(((x1[0].sup, y1), (x2[0].sup, y2)))
            elif isinstance(y1, interval) and isinstance(y2, interval):
                lines.append(((x1, y1[0].inf), (x2, y2[0].inf)))
                lines.append(((x1, y1[0].sup), (x2, y2[0].sup)))
                    
            #lines.append((sorted_intersections[i], sorted_intersections[i + 1]))

    return lines

def subdivide(interval_x, interval_y, intersection_method, tolerance = 1e-5):
    rectangles = [] #All GP intervals
    depth_rectangles = {} #All GP intervals per iteration
    interval_sides = {} #All intersected sides
    lines = [] #Connected lines of approximated curve

    # Initial interval to start BFS
    initial_data = {
        "interval_x": interval_x,
        "interval_y": interval_y,
        "depth": 1
    }
    queue = deque([initial_data])

    while queue:
        current = queue.popleft()
        cur_interval_x, cur_interval_y, cur_depth = current["interval_x"], current["interval_y"], current["depth"]

        #if cur_interval_x[0].inf > -0.9 and cur_interval_x[0].sup < -0.82 and cur_interval_y[0].inf > 0.89 and cur_interval_y[0].sup < 0.97:
        #if cur_interval_x[0].inf > -0.9 and cur_interval_x[0].sup < -0.82 and cur_interval_y[0].inf > 0.92 and cur_interval_y[0].sup < 0.97:
            #print("this one")
        if cur_depth not in depth_rectangles:
            depth_rectangles[cur_depth] = []
            depth_rectangles[cur_depth].extend(rectangles)

        depth_rectangles[cur_depth].append((cur_interval_x, cur_interval_y))

        g_p_status, g_p_option = is_globally_parameterizable(cur_interval_x, cur_interval_y)
        if g_p_status:
            intersections = getIntersectionPoints(cur_interval_x, cur_interval_y, intersection_method, tolerance)
            if not shouldSubdivide(intersections):
                current_lines = buildLines(intersections, g_p_option, intersection_method, tolerance)
                if len(current_lines):
                    lines.extend(current_lines)
                interval_sides[(cur_interval_x, cur_interval_y)] = list(intersections.keys())
                rectangles.append((cur_interval_x, cur_interval_y))
                continue

        subintervals = []

        x_mid = midpoint(cur_interval_x)
        y_mid = midpoint(cur_interval_y)

        match g_p_option:
            case "x":
                subintervals = [
                (cur_interval_x, interval([cur_interval_y[0].inf, y_mid])),
                (cur_interval_x, interval([y_mid, cur_interval_y[0].sup]))
            ]
            case "y":
                subintervals = [
                (interval([cur_interval_x[0].inf, x_mid]), cur_interval_y),
                (interval([x_mid, cur_interval_x[0].sup]), cur_interval_y)
            ]
            case "" | "no root":
                subintervals = [
                (interval([cur_interval_x[0].inf, x_mid]), interval([cur_interval_y[0].inf, y_mid])),
                (interval([x_mid, cur_interval_x[0].sup]), interval([cur_interval_y[0].inf, y_mid])),
                (interval([cur_interval_x[0].inf, x_mid]), interval([y_mid, cur_interval_y[0].sup])),
                (interval([x_mid, cur_interval_x[0].sup]), interval([y_mid, cur_interval_y[0].sup]))
            ]

        for sub_x, sub_y in subintervals:
            g_p_sub_status, g_p_sub_options = is_globally_parameterizable(sub_x, sub_y)
            if g_p_sub_status:
                intersections = getIntersectionPoints(sub_x, sub_y, intersection_method, tolerance)
                if shouldSubdivide(intersections):
                    new_data = {
                            "interval_x": sub_x,
                            "interval_y": sub_y,
                            "depth": cur_depth + 1
                    }
                    queue.append(new_data)
                else:
                    current_lines = buildLines(intersections, g_p_sub_options, intersection_method, tolerance)
                    if len(current_lines):
                        lines.extend(current_lines)
                    interval_sides[(sub_x, sub_y)] = list(intersections.keys())
                    rectangles.append((sub_x, sub_y))

                    if cur_depth + 1 not in depth_rectangles:
                        depth_rectangles[cur_depth + 1] = []
                        depth_rectangles[cur_depth + 1].extend(rectangles)
                    depth_rectangles[cur_depth + 1].append((sub_x, sub_y))
            elif g_p_sub_options != "no root":
                new_data = {
                    "interval_x": sub_x,
                    "interval_y": sub_y,
                    "depth": cur_depth + 1
                }
                queue.append(new_data)

    return rectangles, lines, interval_sides, depth_rectangles

def display(displayOptions, rectangles, lines, interval_sides, depth_rectangles):
    match displayOptions:
        case "final":
            final_plot(rectangles, interval_sides, lines)
        case "depth":
            plot_by_depth(depth_rectangles)

def is_float(string):
    try:
        # float() is a built-in function
        float(string)
        return True
    except ValueError:
        return False
    
def parseInputParameters():
    intersection_method = "newton"
    display_method = "final"
    tolerance = 1e-5

    for argument in sys.argv:
        if is_float(argument):
            tolerance = float(argument)
            continue
        match argument:
            case "depth":
                display_method = "depth"
            case "final":
                display_method = "final"
            case "newton":
                intersection_method = "newton"
            case "binary":
                intersection_method = "binary"
            case "runge":
                intersection_method = "runge"
    return intersection_method, display_method, tolerance

# Define the main function
def main():

    x_interval = interval[-5.1, 5.1]
    y_interval = interval[-5.1, 5.1]
    all_times = []
    all_iterations = []
    all_cache_hits = []
    sum_time = 0
    sum_iterations = 0
    sum_cache_hits = 0
    intersection_method, display_method, tolerance = parseInputParameters()
    global iterations
    global cache_hit_counter
    total_start_time = time.time()
    for i in range(1000):
        start_time = time.time()
        rectangles, lines, intersected_sides, depth_rectangles = subdivide(x_interval, y_interval, intersection_method, tolerance)
        all_times.append(time.time() - start_time)
        all_iterations.append(iterations)
        sum_iterations += iterations
        sum_cache_hits += cache_hit_counter
        all_cache_hits.append(cache_hit_counter)
        cache_hit_counter = 0
        iterations = 0
        sum_time += time.time() - start_time
    print("--- In total took %s seconds ---" % (time.time() - total_start_time))
    print("--- In total made %s iterations ---" % sum_iterations)
    print("--- In total made %s cache hits ---" % sum_cache_hits)

    print("--- On average took %s seconds ---" % (sum_time/len(all_times)))
    print("--- On average made %s iterations ---" % (sum_iterations/len(all_iterations)))
    print("--- On average made %s cache hits ---" % (sum_cache_hits/len(all_cache_hits)))
    #print("iterations = " + str(iterations))
    #print("cache hit = " + str(cache_hit_counter))
    display(display_method, rectangles, lines, intersected_sides, depth_rectangles)
    

main()

