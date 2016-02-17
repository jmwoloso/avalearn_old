# Authors: Jason Wolosonovich
# License: BSD 3 clause
"""
linalg.py:  Set of linear algebra classes and utility functions
            created in the Udacity Linear Algebra Refresher Course.
"""

import numpy as np

class Vector(object):
    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = "Cannot normalize the zero " \
                                       "vector"
    NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG = "No unique orthogonal " \
                                         "component"
    NO_UNIQUE_PARALLEL_COMPONENT_MSG = "No unique parallel component"
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([np.around(x, 30) for x
                                     in coordinates])
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError("The coordinates must be nonempty")

        except TypeError:
            raise TypeError("The coordinates must be an iterable")

    def plus(self, v):
        new_coordinates = [x+y for x,y in zip(self.coordinates,
                                              v.coordinates)]
        return Vector(new_coordinates)

    def minus(self, v):
        new_coordinates = [x-y for x,y in zip(self.coordinates,
                                              v.coordinates)]
        return Vector(new_coordinates)

    def times_scalar(self, c):
        new_coordinates = [c*x for x in self.coordinates]
        return Vector(new_coordinates)

    def magnitude(self):
        squares = [x**2 for x in self.coordinates]
        return np.sqrt(np.sum(squares))

    def normalization(self):
        try:
            magnitude = self.magnitude()
            return self.times_scalar(np.around(1., 30)/magnitude)

        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)

    def dot(self, w):
        #dot_product = mag * mag * cos*theta
        products = [x*y for x, y in zip(self.coordinates,
                                           w.coordinates)]
        return np.sum(products)

    def angle_with(self, w, in_degrees=False):
        try:
            u1 = self.normalization()
            u2 = w.normalization()
            angle_in_radians = np.arccos(u1.dot(u2))

            if in_degrees:
                degrees_per_radian = 180. / np.pi
                return angle_in_radians * degrees_per_radian
            else:
                return angle_in_radians

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZER0_VECTOR_MSG:
                raise Exception("Cannot compute an angle with the "
                                "zero vector")
            else:
                raise(e)

    def is_zero(self, tolerance=1e-10):
        return self.magnitude() < tolerance

    def is_parallel_to(self, w):
        # two vectors are parallel if one is a scalar of the other
        return (self.is_zero() or
                w.is_zero() or
                self.angle_with(w) == 0 or
                self.angle_with(w) == np.pi)

    def is_orthogonal_to(self, w, tolerance=1e-10):
        return np.abs(self.dot(w)) < tolerance

    def component_orthogonal_to(self, b):
        try:
            projection = self.component_parallel_to(b)
            return self.minus(projection)
        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT_MSG:
                raise Exception(
                    self.NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG)
            else:
                raise e

    def component_parallel_to(self, b):
        try:
            u = b.normalization()
            weight = self.dot(u)
            return u.times_scalar(weight)

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e

    def cross(self, v):
        try:
            x_1, y_1, z_1 = self.coordinates
            x_2, y_2, z_2 = v.coordinates
            new_coordinates = [y_1*z_2 - y_2*z_1,
                               -(x_1*z_2 - x_2*z_1),
                               x_1*y_2 - x_2*y_1]

            return Vector(new_coordinates)

        except ValueError as e:
            msg = str(e)
            if msg == 'need more than 2 values to unpack':
                self_embedded_in_R3 = Vector(self.coordinates + ('0',))
                v_embedded_in_R3 = Vector(v.coordinates + ('0',))
                return self_embedded_in_R3.cross(v_embedded_in_R3)
            elif(msg == 'too many values to unpack' or
                 msg == 'need more than 1 value to unpack'):
                raise Exception(self.ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG)
            else:
                raise e

    def area_of_triangle_with(self, v):
        return self.area_of_parallelogram_with(v) / np.around(2., 30)

    def area_of_parallelogram_with(self, v):
        cross_product = self.cross(v)
        return cross_product.magnitude()


    def __str__(self):
        return "Vector: {}".format(self.coordinates)

    def __eq__(self, v):
        return self.coordinates == v.coordinates





