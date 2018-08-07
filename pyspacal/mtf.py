#!/usr/bin/python
"""functions to find the modulation transfer function
"""

import argparse
import imageio
import os
import warnings
import subprocess
import numpy as np
import pandas as pd
from . import utils
from . import camera_data
from scipy import optimize
from matplotlib import pyplot as plt


# these values are taken directly from the paper
PAPER_RGB_TO_LMS = 1e-7 * np.array([[.1267, .2467, -.0450], [.0327, .2904, -.0299],
                                    [.0038, -.0292, .2425]])
LMS_TO_LUM = np.array([433.9441, 275.82, -.0935])


def create_mosaic_mask(img_shape, bayer_mosaic):
    """create mask based on Bayer mosaic of the camera
    
    this is based on the Brainard's lab's SimToolbox's SimCreateMask.m"""
    # First create a small kernel representing the RGB mosaic pattern
    kernels = np.zeros((len(np.unique(bayer_mosaic)), bayer_mosaic.shape[0], bayer_mosaic.shape[1]))
    for i in range(bayer_mosaic.shape[0]):
        for j in range(bayer_mosaic.shape[1]):
            kernels[bayer_mosaic[i, j], i, j] = 1

    # Then repeat the kernel to build a mask to the size  of the image
    mask = []
    for n in range(kernels.shape[0]):
        mask.append(np.tile(kernels[n], (int(np.ceil(img_shape[0] / float(bayer_mosaic.shape[0]))+1), int(np.ceil(img_shape[1] / float(bayer_mosaic.shape[1]))+1))))
    mask = np.dstack(mask).transpose((2, 0, 1))
    if mask.shape[1] < img_shape[0] or mask.shape[2] < img_shape[1]:
        raise Exception('Logic error in computation of mask size')
    mask = mask[:, :img_shape[0], :img_shape[1]]
    return mask, kernels


def demosaic_image(img, mosaic_mask):
    """demosaic the image based on the mosaic mask
    """
    raw_demosaiced_image = np.zeros(mosaic_mask.shape)
    for i in range(mosaic_mask.shape[0]):
        raw_demosaiced_image[i] = img * mosaic_mask[i]
    return raw_demosaiced_image


def _block_avg_helper(img):
    """block average a single channel of the image
    
    note that the image you pass must be 2d"""
    assert img.ndim == 2, "Can only block average 2d images!"
    avged = 0.25*(img[1:,:-1] + img[:-1, 1:] + img[:-1,:-1] + img[1:,1:])
    return avged[::2, ::2]   


def block_avg(raw_demosaiced_img, weights=[4, 2, 4]):
    """block average the raw demosaiced image

    this block average is a weighted average, so we need the weights to use. the defaults are the
    values used by Tkacik et al.
    """
    demosaiced_image = np.zeros((raw_demosaiced_img.shape[0], raw_demosaiced_img.shape[1] // 2,
                                 raw_demosaiced_img.shape[2] // 2))
    for i, w in zip(range(demosaiced_image.shape[0]), weights):
        demosaiced_image[i] = w * _block_avg_helper(raw_demosaiced_img[i])
    return demosaiced_image


def standardizing_constant(iso, f_number, exposure_time, **kwargs):
    """calculate the constant to convert raw to standard RGB values
    
    note that kwargs is ignored, we just use it so we can swallow the extra keys from the 
    metadata dict"""
    return (1000. / iso) * (f_number / 1.8)**2 * (1. / exposure_time)


def standardize_rgb(demosaiced_img, iso, f_number, exposure_time, **kwargs):
    """standardize the rgb values of the demosaiced image

    note that kwargs is ignored, we just use it so we can swallow the extra keys from the 
    metadata dict"""
    return demosaiced_img * standardizing_constant(iso, f_number, exposure_time)


def calculate_rgb_to_lms(s_rgb, s_lms):
    """calculate the matrix that converts from rgb to lms values

    this regresses the two sensitivity matrices against each other to find the 3x3 matrix that best
    converts from RGB to LMS values.
    """
    return np.linalg.lstsq(s_rgb, s_lms)[0].transpose()


def calculate_conversion_matrix_scaling_factor(calc_rgb_to_lms, paper_rgb_to_lms=PAPER_RGB_TO_LMS):
    """calculate the scalar that best matches our calculated conversion matrix to the paper's

    the calculated rgb_to_lms matrix is based on sensitivities from the camspec database, which has
    them in terms of the relative sensitivities (so the max is 1). the paper uses the absolute
    sensitivities and so they're very different. since the values coming off the sensor (and
    persisting through our calculation) are non-normalized, we want to get our conversion matrix on
    approximately that order of magnitude. we do that by finding the scalar value that best matches
    up our conversion matrix with the paper's
    """
    def find_scaling_factor(x):
        return (paper_rgb_to_lms - x*calc_rgb_to_lms).flatten()

    res = optimize.least_squares(find_scaling_factor, 1)    
    return res.x


def luminance_image(standard_rgb_img, rgb_to_lms, scaling_factor=1., lms_to_lum=LMS_TO_LUM):
    """convert the standard rgb image to luminance

    in order for this to all work out:
      - standard_rgb_img.shape must be (3, m, n)
      - rgb_to_lms.shape must be (3, 3)
      - lms_to_lum.shape must be (3,) (that is, a 1d array of length 3)

    scaling_factor: float. how much to scale the rgb_to_lms matrix by. this is used when your
    rgb_to_lms is the calculated version, and so you need to rescale it so it can handle the much
    larger values of the standard_rgb_img (see `calculate_conversion_matrix_scaling_factor` for more
    details)
    """
    # that transpose is to get the dimensions in the right order for matmul. I'm unsure why that's
    # the way it is, but so it goes.
    return np.dot(lms_to_lum, np.matmul(scaling_factor * rgb_to_lms,
                                        standard_rgb_img.transpose(1,0,2)))


def _find_circles(X, grating_edge, white_edge, black_edge):
    """find the distance from optimal we are for our given parameters X

    because of how scipy.optimize works, all the parameters to optimize have to be included in one
    argument, X. Those should be (in order): x0, y0, r_grating, border_ring_width
    """
    x0, y0, r_grating, border_ring_width = X
    vals = []
    for x, y in grating_edge:
        vals.append((x0 - x)**2 + (y0 - y)**2 - r_grating**2)
    for x, y in white_edge:
        vals.append((x0 - x)**2 + (y0 - y)**2 - (r_grating + border_ring_width)**2)
    for x, y in black_edge:
        vals.append((x0 - x)**2 + (y0 - y)**2 - (r_grating + 2*border_ring_width)**2)
    return vals


def _get_initial_guess(circle_ctr, grating_edge, white_edge, black_edge):
    """get an initial guess for the parameters

    this just converts from the values in pts_dict to the parameters we use to optimize.
    """
    rs = []
    for edge in [grating_edge, white_edge, black_edge]:
        if type(edge) is list:
            edge = edge[0]
        rs.append(np.sqrt((circle_ctr[0] - edge[0])**2 + (circle_ctr[1]-edge[1])**2))
    return circle_ctr[0], circle_ctr[1], rs[0], np.mean(rs[1:]) - rs[0]


def find_mask_params(circle_ctr, grating_edge, white_edge, black_edge):
    """find the parameters we can use to define region-defining masks

    given the points provided by the user, we use scipy.optimize to find the center of the circular
    region containing the grating (x0, y0), the diameter of the grating (r_grating), and the width
    of each of the two border rings (border_ring_width). we do this using _find_circles, which
    tries to find those parameters such that we minimize the difference between $(x0 - x)^2 + (y0 -
    y)^2 - r^2$, for the corresponding (x, y) and r for each ring (edge of the grating, edge of the
    white border region, edge of the black border region); note that (x0, y0) is shared for all of
    these.

    this works best when your picture is taken straight-on from the image; the farther you are from
    this ideal, the worse this will perform (because the grating then won't actually fall in a
    circle in the image)

    returns: x0, y0, r_grating, border_ring_width
    """
    res = optimize.least_squares(_find_circles, _get_initial_guess(circle_ctr, grating_edge,
                                                                   white_edge, black_edge),
                                 args=(grating_edge, white_edge, black_edge))
    return res.x


def run_fft(grating_1d):
    """run the fft, returning power, amplitude, fft, and frequencies

    grating_1d should be a 1d array

    this normalizes the grating by its mean and the fft by its length, so that the value of the DC
    component is 1.
    """
    freqs = np.fft.fftshift(np.fft.fftfreq(len(grating_1d),))
    fft = np.fft.fftshift(np.fft.fft(grating_1d / grating_1d.mean()))
    fft /= len(fft)

    power = fft * np.conj(fft)
    amp = np.sqrt(power)

    return power, amp, fft, freqs    


def check_filtering(grating_1d, filtered_grating_1d, contrast):
    """plot
    """
    plt.figure(figsize=(25, 5))
    plt.plot(grating_1d)
    plt.title('1d grating')
    
    plt.figure(figsize=(25, 5))
    plt.plot(filtered_grating_1d)
    plt.title('Filtered fundamental')
    
    print("Contrast: %s" % contrast)

def calculate_contrast(grating_1d, n_phases=20, plot_flag=True):
    """calculate the contrast of the specified grating

    following the procedure used by Tkacik et al, we try several different phase crops of the
    grating, dropping the first range(n_phases) points, and seeing which phase has the highest
    power (across all frequencies). we crop to this phase, then find the frequency that has the
    most power (ignoring the DC component) and filter out all other frequencies. we use this
    filtered grating to compute the Michelson contrast of this filtered grating

    returns: filtered grating, Michelson contrast, and the frequency at which the max power is
    reached
    """
    amps = {}
    for phase in range(n_phases):
        power, amp, fft, freqs = run_fft(grating_1d[phase:])
        power[freqs==0] = 0
        amps[phase] = np.max(amp[np.argwhere(power==np.max(power))])
        
    max_phase = max(amps, key=lambda key: amps[key])
    power, amp, fft, freqs = run_fft(grating_1d[max_phase:])
    
    if plot_flag:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # plotting the fft doesn't make a lot of sense because it's complex-valued
        for ax, data, title in zip(axes, [power, amp], ['power', 'amplitude']):
            ax.stem(freqs[freqs>=0], data[freqs>=0])
            ax.set_title(title)

    power[freqs==0] = 0
    power_argmax = np.argwhere(power==np.max(power))

    if plot_flag:
        print("Max phase crop: %s" % max_phase)
        print("Max frequencies:\n %s\n" % freqs[power_argmax])
        print("Amplitude at max freqs:\n %s\n" % amp[power_argmax])

    filtered_fft = np.zeros(len(fft), dtype=np.complex)
    filtered_fft[np.argwhere(freqs==0)] = fft[np.argwhere(freqs==0)]
    filtered_fft[power_argmax] = fft[power_argmax]

    filtered_grating_1d = np.abs(np.fft.ifft(np.fft.ifftshift(filtered_fft)) * len(fft))
    contrast = (np.max(filtered_grating_1d) - np.min(filtered_grating_1d)) / (np.max(filtered_grating_1d) + np.min(filtered_grating_1d))
    
    if plot_flag:
        check_filtering(grating_1d, filtered_grating_1d, contrast)
    
    # this is the positive of the two frequencies
    return filtered_grating_1d, contrast, freqs[power_argmax][-1]


def main(fname):
    """calculate the contrast, doing all of the steps in between

    see the MTF notebook to see this calculation stepped through one step at a time

    fname: str, the path to the NEF file to analyze
    """
    if not fname.endswith(".NEF"):
        raise Exception("Must pass in the NEF image! %s" % fname)
    if not os.path.exists(fname.replace("NEF", "pgm")):
        subprocess.call(["dcraw", "-4", "-d", fname])
    fname_stem = fname.replace(".NEF", "")
    img, metadata = utils.load_img_with_metadata(fname_stem)
    mask, kernels = create_mosaic_mask(img.shape, camera_data.BAYER_MATRICES[metadata['camera']])
    raw_demosaiced_image = demosaic_image(img, mask)
    demosaiced_image = block_avg(raw_demosaiced_image)
    standard_RGB = standardize_rgb(demosaiced_image, **metadata)
    s_lms = utils.load_cone_fundamental_matrix()
    s_rgb = utils.load_camspec_sensitivity_matrix(camera=metadata['camera'])    
    rgb_to_lms = calculate_rgb_to_lms(s_rgb, s_lms)
    scaling_factor = calculate_conversion_matrix_scaling_factor(rgb_to_lms)
    lum_image = luminance_image(standard_RGB, rgb_to_lms, scaling_factor)
    imageio.imsave(fname_stem + "_lum.png", lum_image)
    try:
        pts_dict = utils.load_pts_dict(metadata['filename'], lum_image.shape)
    except KeyError:
        warnings.warn("Can't find points for %s, please add them!" % fname)
        return None, None, None, None
    else:
        fig = utils.check_pts_dict(lum_image, pts_dict)
        fig.savefig(fname_stem + "_pts.png", dpi=96)
        plt.close(fig)
        x0, y0, r_grating, border_ring_width = find_mask_params(**pts_dict)
        grating_mask, white_mask, black_mask = utils.create_circle_masks(lum_image.shape, x0, y0,
                                                                         r_grating, border_ring_width)
        fig = utils.plot_masked_images(lum_image, [grating_mask, white_mask, black_mask])    
        fig.savefig(fname_stem + "_masks.png")
        plt.close(fig)
        grating_1d = utils.extract_1d_grating(lum_image, grating_mask)
        ring = utils.extract_1d_border(lum_image, white_mask, black_mask, x0, y0)
        filtered_grating_1d, grating_contrast, grating_freq = calculate_contrast(grating_1d, plot_flag=False)
        filtered_ring, ring_contrast, ring_freq = calculate_contrast(ring, plot_flag=False)
        return grating_freq, grating_contrast, ring_freq, ring_contrast


def mtf(fnames, force_run=False):
    """run the entire MTF calculation

    we only re-run this on the images that have not already been analyzed. to over-rule that
    behavior (and run on all specified images), set force_run=True

    fnames: list of strs, the paths to the NEF files to analyze
    """
    grating_freqs, grating_contrasts, ring_freqs, ring_contrasts, content = [], [], [], [], []
    try:
        orig_df = pd.read_csv('mtf.csv')
    except FileNotFoundError:
        analyzed_fnames = []
        orig_df = None
    else:
        analyzed_fnames = orig_df.filenames.unique()
    if not force_run:
        fnames = [f for f in fnames if f not in analyzed_fnames]
    else:
        orig_df = orig_df[~orig_df.filenames.isin(fnames)]
    if fnames:
        print("Analyzing:\n\t%s"%"\n\t".join(fnames))
    else:
        print("No new images to analyze, exiting...")
        return
    for f in fnames:
        gf, gc, rf, rc = main(f)
        if gc is not None:
            grating_freqs.append(np.abs(gf[0]))
            ring_freqs.append(np.abs(rf[0]))
            grating_contrasts.append(gc)
            ring_contrasts.append(rc)
            content.append(camera_data.IMG_INFO[os.path.split(f.replace(".NEF", ""))[-1]])
    df = pd.DataFrame(
        {'grating_frequencies': grating_freqs, 'grating_contrasts': grating_contrasts,
         'ring_frequencies': ring_freqs, 'ring_contrasts': ring_contrasts, 'filenames': fnames,
         'image_content': content})
    tmps = []
    for name in ['grating', 'ring']:
        tmp = df[['image_content', 'filenames', '%s_frequencies' % name, '%s_contrasts' % name]]
        tmp = tmp.rename(columns={'%s_frequencies'%name: 'frequency',
                                  '%s_contrasts'%name: 'contrast'})
        tmp['grating_type'] = name
        tmps.append(tmp)
    df = pd.concat(tmps)
    if orig_df is not None:
        df = pd.concat([orig_df, df])
    df.to_csv("mtf.csv", index=False)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate the MTF using the specified images")
    parser.add_argument("images", nargs='+',
                        help=("What images to calculate contrast for and use to create the MTF"))
    parser.add_argument("--force_run", "-f", action="store_true",
                        help=("Whether to run on all specified images or not. If not passed, we "
                              "skip all images that have already been analyzed"))
    args = vars(parser.parse_args())
    mtf(args['images'], args['force_run'])
