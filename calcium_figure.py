"""
__author__ = Hagai Hargil
"""
from tifffile import imread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import matplotlib.path as mplPath
from scipy.stats import mode
from collections import namedtuple
from matplotlib.ticker import NullFormatter


def get_mask(img, coor_x, coor_y):
    """
    From ROIPOLY
    """
    ny, nx = np.shape(img)
    poly_verts = [(coor_x[0], coor_y[0])]
    for i in range(len(coor_x) - 1, -1, -1):
        poly_verts.append((coor_x[i], coor_y[i]))

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    ROIpath = mplPath.Path(poly_verts)
    grid = ROIpath.contains_points(points).reshape((ny, nx))
    return grid


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def find_brightest_pixels(fname, all_masks, percentile):
    """

    :param fname:
    :param all_masks:
    :return:
    """
    print(f"Loading {fname}...")
    all_data = imread(fname)
    top5_fluo_trace = np.zeros((len(all_masks), all_data.shape[0]))
    bot5_fluo_trace = np.zeros((len(all_masks), all_data.shape[0]))
    if percentile is None:
        pos_perc = None
    else:
        pos_perc = -percentile

    for idx, mask in enumerate(all_masks):
        cur_pix = np.where(mask)
        cur_data = all_data[:, cur_pix[0], cur_pix[1]]
        sorted_pixels = np.sort(cur_data, axis=1)
        top5_fluo_trace[idx, :] = np.mean(sorted_pixels[:, pos_perc:], axis=1)
        bot5_fluo_trace[idx, :] = np.mean(sorted_pixels[:, :percentile], axis=1)
    #     all_pixel_means = np.mean(cur_data, axis=0)
    #     perc95 = np.percentile(all_pixel_means, q=95.)
    #     perc05 = np.percentile(all_pixel_means, q=5.)

    #     relevant_pixels_top = all_pixel_means >= perc95
    #     relevant_pixels_bot = all_pixel_means <= perc05

    #     top5_fluo_trace[idx, :] = np.mean(cur_data[:, relevant_pixels_top], axis=1)
    #     bot5_fluo_trace[idx, :] = np.mean(cur_data[:, relevant_pixels_bot], axis=1)

    plt.figure()
    plt.plot(top5_fluo_trace.T)
    return top5_fluo_trace, bot5_fluo_trace


def gen_list_of_pixels_in_rois(fname, all_masks):
    """
    Create a list, each item in which is an array containing all pixels inside a ROI over time
    :param fname: Raw data
    :param all_masks: ROI masks
    :return: list
    """
    print(f"Reading file {fname}...")
    all_data = imread(fname)
    list_of_rois = []
    for idx, mask in enumerate(all_masks):
        cur_pix = np.where(mask)
        cur_data = all_data[:, cur_pix[0], cur_pix[1]]
        list_of_rois.append(cur_data)

    return list_of_rois


def process_fluo_trace(trace, max_pulses_per_pixel, min_pulses_per_pixel):
    """

    :param trace:
    :param max_pulses_per_pixel:
    :param min_pulses_per_pixel:
    :return:
    """
    lower_boundary = trace / max_pulses_per_pixel
    upper_boundary = trace / min_pulses_per_pixel

    mins_top = np.min(trace, axis=1).reshape((trace.shape[0], 1))
    mins_top = np.tile(mins_top, trace.shape[1])

    df_f_top = trace - mins_top

    mode_top = mode(df_f_top, axis=1)[0].reshape((trace.shape[0], 1))
    mode_top = np.tile(mode_top, trace.shape[1])

    df_f_top = (df_f_top - mode_top) / mode_top
    return df_f_top, lower_boundary, upper_boundary


def compute_df_from_list(list_of_rois, max_pulses_per_pixel, min_pulses_per_pixel):
    """

    :param list_of_rois:
    :param max_pulses_per_pixel:
    :param min_pulses_per_pixel:
    :return:
    """
    Boundaries = namedtuple('Boundaries', ('low', 'high'))
    list_photons_per_pulse = []
    for roi in list_of_rois:
        list_photons_per_pulse.append(Boundaries(roi / max_pulses_per_pixel, roi / min_pulses_per_pixel))

    return list_photons_per_pulse


def plot_all_traces(trace, upper_bound, lower_bound, idx_to_plot=13):
    """

    :param trace:
    :param upper_bound:
    :param lower_bound:
    :param idx_to_plot:
    :return:
    """
    FRAME_RATE = 7.68  # Hz
    fig, ax1 = plt.subplots()
    t = np.arange(trace.shape[1]) / FRAME_RATE
    ax1.plot(t, trace[idx_to_plot, :], 'k', linewidth=1)
    ax1.set_xlabel('Time [sec]', fontsize=14)
    ax1.set_ylabel(r'$\frac{\Delta F}{F}$', color='k', fontsize=16)
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.fill_between(t, upper_bound[idx_to_plot, :],
                     lower_bound[idx_to_plot, :],
                     facecolor=np.array([1., 1., 1.]) * 0.1,
                     alpha=0.2)
    ax2.set_ylabel('# Photons / Pixel', color='gray', fontsize=14)
    ax2.tick_params('y', colors='gray')

    plt.savefig('df_f_photons_pix_26.eps', dpi=1000, transparent=True, format='eps')


def plot_ratio_df_photons(trace, photons, my_data):
    """

    :param trace:
    :param photons:
    :return:
    """
    ravelled_df = np.ravel(trace)
    ravelled_photons = np.ravel(photons)
    pairs = np.column_stack((ravelled_df, ravelled_photons))
    fig, ax = plt.subplots()
    ax.scatter(pairs[:, 0], pairs[:, 1], s=0.5, cmap='gray')
    ax.set_xlabel('dF / F')
    ax.set_ylabel('# Photons / Pixel')
    return pairs, fig, ax


def correlation_of_ratio_to_loc(pairs, my_data, data_len, fig, ax):
    """

    :param pairs:
    :param my_data:
    :param data_len:
    :param fig:
    :param ax:
    :return:
    """

    nullfmt = NullFormatter()  # no labels
    X_CENTER = 512
    Y_CENTER = 512

    centers = []
    for roi in my_data['rois']:
        x_dist = (np.mean(roi[0]) - X_CENTER) ** 2
        y_dist = (np.mean(roi[1]) - Y_CENTER) ** 2
        centers.append(np.sqrt(x_dist + y_dist))
    centers = np.array(centers)

    distance_vector = np.zeros((pairs.shape[0]))
    for idx, center in enumerate(centers):
        start_idx = idx * data_len
        end_idx = idx * data_len + data_len
        distance_vector[start_idx:end_idx] = center

    ## Plot
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(pairs[:, 0])), np.max(np.fabs(pairs[:, 1]))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(distance_vector, bins=bins)
    axHisty.hist(pairs[:, 0], bins=bins, orientation='horizontal')

    axHistx.set_xlim(ax.get_xlim())
    axHisty.set_ylim(ax.get_ylim())


class Plotting:
    pass


def main():
    """
    Main
    :return:
    """
    PIXEL_CLOCK = 44e-9  # seconds
    LASER_PERIOD = 12.5e-9  # seconds
    max_pulses_per_pixel = np.ceil(PIXEL_CLOCK/LASER_PERIOD)
    min_pulses_per_pixel = np.floor(PIXEL_CLOCK/LASER_PERIOD)

    my_data = np.load(
        r'X:\Hagai\Multiscaler\27-9-17\For article\Calcium\vessel_neurons_analysis_stop1_pmt1_stop2_lines_unidir_power_48p5_gain_850_0081.npz')
    my_data = my_data['arr_0'].item()

    all_masks = []
    for roi in my_data['rois']:
        all_masks.append(get_mask(my_data['img_neuron'], roi[0], roi[1]))

    fname = r'X:\Hagai\Multiscaler\27-9-17\For article\Calcium\stop1_pmt1_stop2_lines_unidir_power_48p5_gain_850_008.tif'
    top5_fluo_trace, bot5_fluo_trace = \
        find_brightest_pixels(fname, all_masks, percentile=None)

    df_f_top, lower_boundary_brightest_pixels, \
        upper_boundary_brightest_pixels = process_fluo_trace(
        top5_fluo_trace, max_pulses_per_pixel, min_pulses_per_pixel)

    # list_of_rois = gen_list_of_pixels_in_rois(fname, all_masks)
    # list_photons_per_pulse = compute_df_from_list(list_of_rois, max_pulses_per_pixel,
    #                                               min_pulses_per_pixel)


    plot_all_traces(df_f_top, upper_boundary_brightest_pixels, lower_boundary_brightest_pixels,
                    idx_to_plot=26)  # second idx used in the article is 15

    pairs, fig, ax = plot_ratio_df_photons(df_f_top, upper_boundary_brightest_pixels, my_data)
    _, fig2, ax2 = plot_ratio_df_photons(df_f_top, upper_boundary_brightest_pixels, my_data)
    correlation_of_ratio_to_loc(pairs, my_data, df_f_top.shape[1], fig, ax)

    return df_f_top

if __name__ == '__main__':
    df = main()
