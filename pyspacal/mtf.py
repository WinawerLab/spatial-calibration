#!/usr/bin/python
"""functions to find the modulation transfer function
"""

import argparse
import imageio
import itertools
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

    amp = np.abs(fft)

    return amp, fft, freqs    


def check_filtering(grating_1d, filtered_grating_1d, normalized_contrast):
    """plot
    """
    plt.figure(figsize=(25, 5))
    plt.plot(grating_1d)
    plt.title('1d grating')
    
    plt.figure(figsize=(25, 5))
    plt.plot(filtered_grating_1d)
    plt.title('Filtered fundamental')
    
    print("Square-wave contrast: %s" % normalized_contrast)

def calculate_contrast(grating_1d, n_phases=20, plot_flag=True):
    """calculate the contrast of the specified grating

    following the procedure used by Tkacik et al, we try several different phase crops of the
    grating, dropping the first range(n_phases) points, and seeing which phase has the highest
    power (across all frequencies). we crop to this phase, then find the frequency that has the
    most power (ignoring the DC component) and filter out all other frequencies. we use this
    filtered grating to compute the Michelson contrast of this filtered grating

    returns: filtered grating, contrast of the fundamental, normalized contrast (of the
    square-wave), and the frequency at which the max power is reached
    """
    amps = {}
    for phase in range(n_phases):
        amp, fft, freqs = run_fft(grating_1d[phase:])
        amp[freqs==0] = 0
        amps[phase] = np.max(amp)
        
    max_phase = max(amps, key=lambda key: amps[key])
    amp, fft, freqs = run_fft(grating_1d[max_phase:])
    
    # since this comes from a real signal, we know the fft is symmetric (if it's not exactly
    # symmetric for some reason, that's because of a precision error), so we just drop the negative
    # frequencies and double the positive ones (except the DC and highest frequency components,
    # which correspond to each other). we do this for the amplitude, since that's what we'll use to
    # get the contrast, but not for the fft, since we use that to reconstruct the filtered signal.
    amp = amp[freqs>=0]
    amp[1:-1] = 2*amp[1:-1]

    if plot_flag:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.stem(freqs[freqs>=0], amp)
        ax.set_title('Amplitude')

    amp[freqs[freqs>=0]==0] = 0
    amp_argmax = np.argmax(amp)
    contrast = amp[amp_argmax]
    # contrast is the contrast of the fundamental; in order to get the contrast of the square-wave
    # it came from, we have to multiply it by pi / 4, see
    # http://mathworld.wolfram.com/FourierSeriesSquareWave.html for an explanation
    normalized_contrast = contrast * np.pi / 4

    freq_max = freqs[freqs>=0][amp_argmax]
    freq_argmax = np.array([np.argwhere(freqs==freq_max)[0],
                            np.argwhere(freqs==-freq_max)[0]])

    if plot_flag:
        print("Max phase crop: %s" % max_phase)
        print("Max frequency:\n %s\n" % freq_max)
        print("Amplitude at max frequency:\n %s\n" % amp[amp_argmax])

    filtered_fft = np.zeros(len(fft), dtype=np.complex)
    filtered_fft[np.argwhere(freqs==0)] = fft[np.argwhere(freqs==0)]
    filtered_fft[freq_argmax] = fft[freq_argmax]

    filtered_grating_1d = np.fft.ifft(np.fft.ifftshift(filtered_fft)) * len(fft)
    # if we did everything right, this should have zero imaginary components
    assert np.allclose(np.imag(filtered_grating_1d), np.zeros(len(fft))), "Something went wrong, the reconstructed grating is complex!"
    filtered_grating_1d = np.real(filtered_grating_1d) * grating_1d.mean()

    if plot_flag:
        check_filtering(grating_1d, filtered_grating_1d, normalized_contrast)
    
    return filtered_grating_1d, contrast, normalized_contrast, freq_max


def main(fname, preprocess_type):
    """calculate the contrast, doing all of the steps in between

    see the MTF notebook to see this calculation stepped through one step at a time

    fname: str, the path to the NEF file to analyze
    """
    if not fname.endswith(".NEF"):
        raise Exception("Must pass in the NEF image! %s" % fname)
    save_stem = os.path.splitext(fname.replace("raw", os.path.join("%s", "{preprocess_type}")))[0]
    save_stem = save_stem.format(preprocess_type=preprocess_type)
    for deriv in ['luminance', 'mask_check', 'preprocessed']:
        if not os.path.isdir(os.path.dirname(save_stem % deriv)):
            os.makedirs(os.path.dirname(save_stem % deriv))
    utils.preprocess_image(fname, preprocess_type)
    img, metadata = utils.load_img_with_metadata(fname, preprocess_type)
    if img.ndim == 2:
        mask, kernels = create_mosaic_mask(img.shape, camera_data.BAYER_MATRICES[metadata['camera']])
        raw_demosaiced_image = demosaic_image(img, mask)
        demosaiced_image = block_avg(raw_demosaiced_image)
    else:
        # we need to rearrange the axes so that RGB is on the first axis.
        demosaiced_image = img.transpose((2, 0, 1))
    standard_RGB = standardize_rgb(demosaiced_image, **metadata)
    s_lms = utils.load_cone_fundamental_matrix()
    s_rgb = utils.load_camspec_sensitivity_matrix(camera=metadata['camera'])    
    rgb_to_lms = calculate_rgb_to_lms(s_rgb, s_lms)
    scaling_factor = calculate_conversion_matrix_scaling_factor(rgb_to_lms)
    lum_image = luminance_image(standard_RGB, rgb_to_lms, scaling_factor)
    imageio.imsave(save_stem % 'luminance' + "_lum.png", lum_image)
    try:
        pts_dict = utils.load_pts_dict(metadata['filename'], lum_image.shape)
    except KeyError:
        warnings.warn("Can't find points for %s, please add them!" % fname)
        return None, None, None, None, None, None, None, None
    else:
        fig = utils.check_pts_dict(lum_image, pts_dict)
        fig.savefig(save_stem % 'mask_check' + "_pts.png", dpi=96)
        plt.close(fig)
        x0, y0, r_grating, border_ring_width = find_mask_params(**pts_dict)
        grating_mask, white_mask, black_mask = utils.create_circle_masks(lum_image.shape, x0, y0,
                                                                         r_grating, border_ring_width)
        fig = utils.plot_masked_images(lum_image, [grating_mask, white_mask, black_mask])    
        fig.savefig(save_stem % 'mask_check' + "_masks.png")
        plt.close(fig)
        grating_1d = utils.extract_1d_grating(lum_image, grating_mask)
        ring = utils.extract_1d_border(lum_image, white_mask, black_mask, x0, y0)
        _, _, grating_contrast, grating_freq = calculate_contrast(grating_1d, plot_flag=False)
        _, _, ring_contrast, ring_freq = calculate_contrast(ring, plot_flag=False)
        return (grating_freq, grating_contrast, ring_freq, ring_contrast, metadata,
                demosaiced_image, standard_RGB, lum_image)


def mtf(fnames, force_run=False):
    """run the entire MTF calculation

    we only re-run this on the images that have not already been analyzed. to over-rule that
    behavior (and run on all specified images), set force_run=True

    fnames: list of strs, the paths to the NEF files to analyze
    """
    if type(fnames) is not list:
        fnames = [fnames]
    # construct the list of tuples (raw_image_filename, preproc_type) to analyze. we make it a list
    # so we can iterate through it multiple times.
    tuples_to_analyze = list(itertools.product(fnames, ['no_demosaic', 'nikon_demosaic',
                                                        'dcraw_vng_demosaic', 'dcraw_ahd_demosaic']))
    grating_freqs, grating_contrasts, ring_freqs, ring_contrasts = [], [], [], []
    # context is what the image was presented on, content is how many cycles are in the image
    content, context = [], []
    # we also want to keep some information about our calculated luminance and other images
    lum_mean, lum_min, lum_max = [], [], []
    demosaic_mean, demosaic_min, demosaic_max = [], [], []
    std_mean, std_min, std_max = [], [], []
    # and the metadata
    iso, f_number, exposure_time, preprocess_types = [], [], [], []
    try:
        orig_df = pd.read_csv('mtf.csv')
    except FileNotFoundError:
        orig_df = pd.DataFrame({'filenames': [], 'preprocess_type': []})
    if not force_run:
        tmp = []
        for f, preproc in tuples_to_analyze:
            # then this exact pair is not in orig_df and so we should analyze it. we use isin
            # instead of == because isin will work for the emptydataframe as well
            if orig_df[(orig_df.filenames.isin([f])) & (orig_df.preprocess_type.isin([preproc]))].empty:
                tmp.append((f, preproc))
        tuples_to_analyze= tmp
    else:
        # iterate through the tuples to analyze and drop all of them from orig_df
        for f, preproc in tuples_to_analyze:
            idx = orig_df[(orig_df.filenames.isin([f])) & (orig_df.preprocess_type.isin([preproc]))].index
            orig_df.drop(idx)
    if tuples_to_analyze:
        print("Analyzing:\n\t%s"%"\n\t".join([", ".join(tup) for tup in tuples_to_analyze]))
    else:
        print("No new images to analyze, exiting...")
        return
    fnames = []
    for f, preproc in tuples_to_analyze:
        print("Analyzing %s with preproc method %s" % (f, preproc))
        gf, gc, rf, rc, meta, demosaic, std, lum = main(f, preproc)
        if gc is not None:
            grating_freqs.append(np.abs(gf))
            ring_freqs.append(np.abs(rf))
            grating_contrasts.append(gc)
            ring_contrasts.append(rc)
            content.append(camera_data.IMG_INFO[os.path.split(f.replace(".NEF", ""))[-1]][1])
            context.append(camera_data.IMG_INFO[os.path.split(f.replace(".NEF", ""))[-1]][0])
            demosaic_mean.append(demosaic.mean())
            demosaic_min.append(demosaic.min())
            demosaic_max.append(demosaic.max())
            std_mean.append(std.mean())
            std_min.append(std.min())
            std_max.append(std.max())
            lum_mean.append(lum.mean())
            lum_min.append(lum.min())
            lum_max.append(lum.max())
            iso.append(meta['iso'])
            f_number.append(meta['f_number'])
            exposure_time.append(meta['exposure_time'])
            preprocess_types.append(preproc)
            fnames.append(f)
    # is there a better way to construct this dataframe? almost certainly, but this does what I
    # want it to do
    df = pd.DataFrame(
        {'grating_frequencies': grating_freqs, 'grating_contrasts': grating_contrasts,
         'ring_frequencies': ring_freqs, 'ring_contrasts': ring_contrasts, 'filenames': fnames,
         'image_content': content, 'image_context': context, 'luminance_mean': lum_mean,
         'luminance_min': lum_min, 'luminance_max': lum_max, 'std_RGB_mean': std_mean,
         'std_RGB_min': std_min, 'std_RGB_max': std_max, 'iso': iso, 'f_number': f_number,
         'exposure_time': exposure_time, 'demosaiced_mean': demosaic_mean,
         'demosaiced_min': demosaic_min, 'demosaiced_max': demosaic_max,
         'preprocess_type': preprocess_types})
    tmps = []
    for name in ['grating', 'ring']:
        tmp = df[['image_content', 'image_context', 'iso', 'f_number', 'exposure_time',
                  'luminance_mean', 'luminance_min', 'luminance_max', 'demosaiced_mean',
                  'demosaiced_min', 'demosaiced_max', 'std_RGB_mean', 'std_RGB_min', 'std_RGB_max',
                  'filenames', '%s_frequencies' % name, '%s_contrasts' % name, 'preprocess_type']]
        tmp = tmp.rename(columns={'%s_frequencies'%name: 'frequency',
                                  '%s_contrasts'%name: 'contrast'})
        tmp['grating_type'] = name
        tmps.append(tmp)
    df = pd.concat(tmps)
    if orig_df is not None:
        df = pd.concat([orig_df, df])
    df = df.reset_index()
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
