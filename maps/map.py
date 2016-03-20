import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import colorConverter
import matplotlib as mpl


def load_map(file_location):
    """
    Loads the map file from the specified location. Returns a numpy array
    :type file_location: str
    :param file_location: The location of the map file. Loads a png grayscale file.

    :return: numpy.ndarray
    """

    def rgb2gray(rgb):
        gray = np.dot(rgb[..., :3], [0.2989, 0.587, 0.114])
        gray[gray < 0.1] = 0.
        gray[gray > 0.1] = 1.
        return gray

    img = mpimg.imread(file_location)

    # invert gray so that free space is 0 and obstances are 1
    grid_init = 1 - rgb2gray(img)

    # set the x-y axes in the convential place
    return np.asarray(np.transpose(grid_init), dtype=np.float64)


class Point:
    """
    Data structure for dealing with (x,y) corrdinates
    """

    def __init__(self, *args):
        """
        :param args: ether input x,y or (x,y).
        :return:
        """
        if len(args) is 1:
            self.x, self.y = args[0]
        else:
            self.x, self.y = args

    def __getitem__(self, index):
        retval = None
        if index is 0:
            retval = self.x
        elif index is 1:
            retval = self.y
        return retval

    def __setitem__(self, key, value):
        if key is 0:
            self.x = value
        elif key is 1:
            self.y = value

    def distance(self, p):
        diff_x, diff_y = self.x - p.x, self.y - p.y
        return math.sqrt(pow(diff_x, 2) + pow(diff_y, 2))

    def toArray(self):
        return np.array([self.x, self.y])

    def toTuple(self):
        return (self.x, self.y)

    def __hash__(self):
        return hash((self.x, self.y))


class MAP2D:
    def __init__(self, window_x, window_y):
        self.set_window_bounds(window_x, window_y)

    def load_map(self, file_location):
        self.filename = file_location
        self.map = load_map(file_location)
        self.alt_map_attrib = np.zeros(self.map.shape, dtype=np.float64)

    def get_bounds(self):
        """
        Return the min and max values for the map indicies
        :rtype : dict
        :return : Dictionary with keys 'x' and 'y'
        """
        return {'x': [0, self.map.shape[0] - 1],
                'y': [0, self.map.shape[1] - 1]}

    def show_map(self):
        """
        Displays the map. Free space is white and obstacles are black. Note that this flips the actual 0/1 values stored in map
        """
        window = np.zeros((2,) + self.map.shape, dtype=np.float64)
        window[0, :, :] = self.map[:, :]
        window[1, :, :] = self.alt_map_attrib[:, :]
        self.show_window(window)

    def _get_valid_bounds(self, x, y):
        """
        Ugly ass function. Ignore if possible
        """
        assert (isinstance(x, int))
        assert (isinstance(y, int))

        dim_x = int(self.window_x / 2)
        dim_y = int(self.window_y / 2)

        min_window_x, min_window_y = 0, 0
        max_window_x, max_window_y = self.window_x, self.window_y

        min_x, min_y = x - dim_x, y - dim_y
        max_x, max_y = x + dim_x + 1, y + dim_y + 1
        if min_x < 0:
            min_x = 0
            min_window_x = abs(x - dim_x)
        if max_x > self.map.shape[0]:
            max_window_x -= (max_x - self.map.shape[0])
            max_x = self.map.shape[0]
        if min_y < 0:
            min_y = 0
            min_window_y = abs(y - dim_y)
        if max_y > self.map.shape[1]:
            max_window_y -= (max_y - self.map.shape[1])
            max_y = self.map.shape[1]

        slice_window = [slice(min_window_x, max_window_x), slice(min_window_y, max_window_y)]
        slice_map = [slice(min_x, max_x), slice(min_y, max_y)]

        return slice_window, slice_map

    def is_collision(self, x, y):
        return self.map[x][y] > 0.5

    def raycast(self, p1, p2, check_dist=0.5):
        """
        Check if there is a collision free ray between the two provided points

        :type p1: Point
        :param p1: Start point

        :type p2: Point
        :param p2: End point

        :type check_dist: FloatX
        :param check_dist: The minimum distance to travel before checking for collision

        :return collision: bool
        """
        # compute vector from one point to the other and make it unit size
        v1, v2 = p1.toArray(), p2.toArray()
        v3 = v2 - v1
        v3 = v3 / np.linalg.norm(v3)

        # Calculate the number of steps needed to be checked
        # check every check_dist pixels traveled
        n_steps = int(math.ceil(p1.distance(p2) / check_dist))
        ray = v1

        for i in xrange(n_steps):
            ray = ray + check_dist * v3

            if self.is_collision(ray[0], ray[1]):
                return True

        return False

    def set_window_bounds(self, window_x, window_y):
        """
        :type window_x: int
        :param window_x: Size of the window around x in pixels
        :type window_y: int
        :param window_y: size of the window around y in pixels
        :return:
        """
        assert (isinstance(window_x, int))
        assert (isinstance(window_y, int))
        self.window_x = window_x
        self.window_y = window_y

    def get_window(self, x, y):
        """
        Returns a subwindow surrounding the point provided. Includes obstacle and alt var channels

        :type x: int
        :param x: X-coordinate
        :type y: int
        :param y: Y-coordinate

        :return window: grid of surrounding window
        """
        # Dimension might extend outside of map. Fill in with obstacles for now
        # Might consider replacing unknown areas with values of 0.5 at some point
        slice_window, slice_map = self._get_valid_bounds(x, y)
        window = np.ones((2, self.window_x, self.window_y), dtype=np.float64)

        slice_window_0 = [0]
        slice_window_1 = [1]
        slice_window_0.extend(slice_window)
        slice_window_1.extend(slice_window)

        window[slice_window_0] = self.map[slice_map]
        window[slice_window_1] = self.alt_map_attrib[slice_map]

        return window


    def show_window(self, subwindow):
        obstable_img = np.transpose(subwindow[0, :, :])
        alt_var_img = np.transpose(subwindow[1, :, :])

        # generate the colors for your colormap

        color1 = colorConverter.to_rgba('white')
        color2 = colorConverter.to_rgba('blue')

        # make the colormaps
        cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', ['white', 'black'], 256)
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', [color1, color2], 256)

        cmap2._init()  # create the _lut array, with rgba values

        # create your alpha array and fill the colormap with them.
        # here it is progressive, but you can create whathever you want
        alphas = np.linspace(0., 1.0, cmap2.N + 3)
        cmap2._lut[:, -1] = alphas

        plt.figure()
        img3 = plt.imshow(obstable_img, interpolation='none', vmin=0, vmax=1, cmap=cmap1, origin='lower')
        plt.hold(True)
        img2 = plt.imshow(alt_var_img, interpolation='none', vmin=0, vmax=1, cmap=cmap2, origin='lower')
        plt.colorbar()
        plt.hold(False)
        plt.show()


if __name__ == '__main__':
    file_location = './maps/squares.png'
    window_size = 83

    m = MAP2D(window_y=window_size, window_x=window_size)
    m.load_map(file_location)

    w = m.get_window(250, 250)
    m.show_window(w)

    p1 = Point(200, 250)
    p2 = Point(450, 400)
    print 'Raycast outcome : '
    print m.raycast(p1, p2)

    m.show_map()
