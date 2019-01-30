#!/usr/bin/python
"""functions to find the modulation transfer function
"""

import argparse
import imageio
import itertools
import os
import warnings
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
    kernels = np.zeros((len(np.unique(bayer_mosaic)), bayer_mosaic.shape[0],
                        bayer_mosaic.shape[1]))
    for i in range(bayer_mosaic.shape[0]):
        for j in range(bayer_mosaic.shape[1]):
            kernels[bayer_mosaic[i, j], i, j] = 1

    # Then repeat the kernel to build a mask to the size  of the image
    mask = []
    for n in range(kernels.shape[0]):
        mask.append(np.tile(kernels[n],
                            (int(np.ceil(img_shape[0] / float(bayer_mosaic.shape[0]))+1),
                             int(np.ceil(img_shape[1] / float(bayer_mosaic.shape[1]))+1))))
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
    avged = 0.25*(img[1:, :-1] + img[:-1, 1:] + img[:-1, :-1] + img[1:, 1:])
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

    note that kwargs is ignored, we just use it so we can swallow the extra keys from the metadata
    dict

    """
    return (1000. / iso) * (f_number / 1.8)**2 * (1. / exposure_time)


def standardize_rgb(demosaiced_img, iso, f_number, exposure_time, **kwargs):
    """standardize the rgb values of the demosaiced image

    note that kwargs is ignored, we just use it so we can swallow the extra keys from the metadata
    dict

    """
    return demosaiced_img * standardizing_constant(iso, f_number, exposure_time)


def calculate_rgb_to_lms(s_rgb, s_lms):
    """calculate the matrix that converts from rgb to lms values

    this regresses the two sensitivity matrices against each other to find the 3x3 matrix that best
    converts from RGB to LMS values.
    """
    return np.linalg.lstsq(s_rgb, s_lms, rcond=None)[0].transpose()


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
    larger values of the standard_rgb_img (see `calculate_conversion_matrix_scaling_factor` for
    more details)

    """
    # that transpose is to get the dimensions in the right order for matmul. I'm unsure why that's
    # the way it is, but so it goes.
    return np.dot(lms_to_lum, np.matmul(scaling_factor * rgb_to_lms,
                                        standard_rgb_img.transpose(1, 0, 2)))


def _find_squares(X, grating_edge, white_edge, black_edge):
    """find the distance from optimal we are for our given parameters X

    because of how scipy.optimize works, all the parameters to optimize have to be included in one
    argument, X. Those should be (in order): x0, y0, r_grating, border_ring_width, angle (should be
    in degrees, within [-45, 45))

    """
    x0, y0, r_grating, border_ring_width, angle = X
    vals = []
    if angle < -45 or angle >= 45:
        raise Exception("angle must lie within [-45, 45)")
    # when the square is rotated, we don't compare it to the radius, but the radius scaled by an
    # amount based on the rotation. we use trig to figure that scale factor out (draw out to
    # convince yourself): as long as the angle lies between -45 and 45 degrees, the line from the
    # center to the edge that forms a right angle (which is equal to the radius), the line to the
    # edge that goes through our point x, y, and the edge form a right triangle, and so we divide r
    # by cos(angle) in order to find the length fo the line that goes to the edge through x, y.
    r_rescale = np.cos(np.deg2rad(angle))
    for x, y in grating_edge:
        vals.append(utils._square_eqt(x, y, x0, y0, angle) - (r_grating / r_rescale))
    for x, y in white_edge:
        vals.append(utils._square_eqt(x, y, x0, y0, angle) -
                    ((r_grating + border_ring_width) / r_rescale))
    for x, y in black_edge:
        vals.append(utils._square_eqt(x, y, x0, y0, angle) -
                    ((r_grating + 2*border_ring_width) / r_rescale))
    return vals


def _get_initial_guess(square_ctr, grating_edge, white_edge, black_edge):
    """get an initial guess for the parameters

    this just converts from the values in pts_dict to the parameters we use to optimize. we always
    guess 0 for the initial angle

    """
    rs = []
    for edge in [grating_edge, white_edge, black_edge]:
        if type(edge) is list:
            edge = edge[0]
        rs.append(utils._square_eqt(edge[0], edge[1], square_ctr[0], square_ctr[1], 0))
    return square_ctr[0], square_ctr[1], rs[0], np.mean(rs[1:]) - rs[0], 0


def find_mask_params(square_ctr, grating_edge, white_edge, black_edge):
    """find the parameters we can use to define region-defining masks

    given the points provided by the user, we use scipy.optimize to find the center of the circular
    region containing the grating (x0, y0), the diameter of the grating (r_grating), and the width
    of each of the two border rings (border_ring_width). we do this using _find_squares, which
    tries to find those parameters such that we minimize the difference between $(x0 - x)^2 + (y0 -
    y)^2 - r^2$, for the corresponding (x, y) and r for each ring (edge of the grating, edge of the
    white border region, edge of the black border region); note that (x0, y0) is shared for all of
    these.

    this works best when your picture is taken straight-on from the image; the farther you are from
    this ideal, the worse this will perform (because the grating then won't actually fall in a
    square in the image)

    returns: x0, y0, r_grating, border_ring_width, angle
    """
    res = optimize.least_squares(_find_squares, _get_initial_guess(square_ctr, grating_edge,
                                                                   white_edge, black_edge),
                                 args=(grating_edge, white_edge, black_edge),
                                 bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0],
                                         [np.inf, np.inf, np.inf, np.inf, 90]))
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


def fourier_contrast(grating_1d, n_phases=20, plot_flag=True):
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
        amp[freqs == 0] = 0
        amps[phase] = np.max(amp)

    max_phase = max(amps, key=lambda key: amps[key])
    amp, fft, freqs = run_fft(grating_1d[max_phase:])

    # since this comes from a real signal, we know the fft is symmetric (if it's not exactly
    # symmetric for some reason, that's because of a precision error), so we just drop the negative
    # frequencies and double the positive ones (except the DC and highest frequency components,
    # which correspond to each other). we do this for the amplitude, since that's what we'll use to
    # get the contrast, but not for the fft, since we use that to reconstruct the filtered signal.
    amp = amp[freqs >= 0]
    amp[1:-1] = 2*amp[1:-1]

    if plot_flag:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.stem(freqs[freqs >= 0], amp)
        ax.set_title('Amplitude')

    amp[freqs[freqs >= 0] == 0] = 0
    amp_argmax = np.argmax(amp)
    contrast = amp[amp_argmax]
    # contrast is the contrast of the fundamental; in order to get the contrast of the square-wave
    # it came from, we have to multiply it by pi / 4, see
    # http://mathworld.wolfram.com/FourierSeriesSquareWave.html for an explanation
    normalized_contrast = contrast * np.pi / 4

    freq_max = freqs[freqs >= 0][amp_argmax]
    freq_argmax = np.array([np.argwhere(freqs == freq_max)[0],
                            np.argwhere(freqs == -freq_max)[0]])

    if plot_flag:
        print("Max phase crop: %s" % max_phase)
        print("Max frequency:\n %s\n" % freq_max)
        print("Amplitude at max frequency:\n %s\n" % amp[amp_argmax])

    filtered_fft = np.zeros(len(fft), dtype=np.complex)
    filtered_fft[np.argwhere(freqs == 0)] = fft[np.argwhere(freqs == 0)]
    filtered_fft[freq_argmax] = fft[freq_argmax]

    filtered_grating_1d = np.fft.ifft(np.fft.ifftshift(filtered_fft)) * len(fft)
    # if we did everything right, this should have zero imaginary components
    assert np.allclose(np.imag(filtered_grating_1d), np.zeros(len(fft))), "Something went wrong, the reconstructed grating is complex!"
    filtered_grating_1d = np.real(filtered_grating_1d) * grating_1d.mean()

    if plot_flag:
        check_filtering(grating_1d, filtered_grating_1d, normalized_contrast)

    return filtered_grating_1d, contrast, normalized_contrast, freq_max


def rms_contrast(image):
    """calculate the RMS contrast of the specified image

    RMS contrast is the standard deviation of the values in the image divided by the mean of the
    image: $\frac{\sqrt{\frac{\sum{I_i - \bar{I}}}{n}}}{\bar{I}}$. note that, when you're getting
    contrast of some patch / subset of the image, you should just pass that part of the image; the
    mean comes from the same local patch (rather than being the mean of the whole image). also note
    that this is not bounded between 0 and 1; for example, the RMS contrast of a delta function is
    infinite. and note that it will be 1 when there are equal number of white and black pixels, but
    if, for example, there are more black pixels, RMS will be less than 1.

    returns: RMS contrast
    """
    return np.sqrt(np.mean(np.square(image - np.mean(image)))) / np.mean(image)


def _calc_freq(image_1d, n_cycles):
    return n_cycles / len(image_1d)


def get_frequency(image, metadata, grating_mask, white_mask, black_mask, x0, y0, angle):
    """get the frequency of the two segments of the passed in image (grating and border)

    we cheat with this, using the fact that we know the number of cycles in the grating (from the
    image metadata) and the border (which is 1), so we only need to find the length of the signals
    in order to get the frequency
    """
    grating_1d = utils.extract_1d_grating(image, grating_mask, metadata['grating_direction'],
                                          angle)
    border_1d = utils.extract_1d_border(image, white_mask, black_mask, x0, y0, angle)
    # this converts a string of the format "32 cyc/image" into the integer 32
    grating_freq = _calc_freq(grating_1d, int(metadata['image_content'].split(' ')[0]))
    border_freq = _calc_freq(border_1d, 1)
    return grating_freq, border_freq


def create_luminance_image(fname, preprocess_type):
    """creates the luminance image, doing everything from the beginning to that

    fname: str, the path to the NEF file to analyze

    returns luminance image
    """
    if not fname.endswith(".NEF"):
        raise Exception("Must pass in the NEF image! %s" % fname)
    save_stem = os.path.splitext(fname.replace("raw", os.path.join("%s", "{preprocess_type}")))[0]
    save_stem = save_stem.format(preprocess_type=preprocess_type)
    for deriv in ['luminance', 'mask_check', 'preprocessed', 'corrected_luminance']:
        if not os.path.isdir(os.path.dirname(save_stem % deriv)):
            os.makedirs(os.path.dirname(save_stem % deriv))
    utils.preprocess_image(fname, preprocess_type)
    img, metadata = utils.load_img_with_metadata(fname, preprocess_type)
    if img.ndim == 2:
        mask, kernels = create_mosaic_mask(img.shape,
                                           camera_data.BAYER_MATRICES[metadata['camera']])
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
    return lum_image, metadata, save_stem, demosaiced_image, standard_RGB


def main(fname, preprocess_type, blank_lum_image=None):
    """calculate the contrast, doing all of the steps in between

    see the MTF notebook to see this calculation stepped through one step at a time

    fname: str, the path to the NEF file to analyze

    blank_lum_image: 2d numpy array, optional. If set, will divide the luminance image created
    during this function element-wise by this array. Should do this in order to counter-act any
    variations in luminance across the screen (for example, if the luminance is highest in the
    center of the screen)
    """
    lum_image, metadata, save_stem, demosaiced_image, standard_RGB = create_luminance_image(fname, preprocess_type)
    # we'll have 4 rows per image: 2 by 2 coming from (grating, border) by (rms, fourier). that is,
    # which part of the image we're giving the contrast for and how we calculated it.
    metadata.update({'frequency': None, 'contrast': None, 'lum_corrected_contrast': None,
                     'grating_type': ['grating', 'grating', 'border', 'border'],
                     'contrast_type': ['rms', 'fourier', 'rms', 'fourier']})
    df = pd.DataFrame(metadata)
    funcs = zip(['mean', 'min', 'max'], [np.mean, np.min, np.max])
    imgs = zip(['luminance', 'demosaiced', 'std_RGB'], [lum_image, demosaiced_image, standard_RGB])
    for (img_name, img), (func_name, func) in itertools.product(imgs, funcs):
        df['%s_%s' % (img_name, func_name)] = func(img)
    df.set_index(['grating_type', 'contrast_type'], inplace=True)
    try:
        pts_dict = utils.load_pts_dict(metadata['filename'], lum_image.shape)
    except KeyError:
        warnings.warn("Can't find points for %s, please add them!" % fname)
        return None, None, None, None
    else:
        fig = utils.check_pts_dict(lum_image, pts_dict)
        fig.savefig(save_stem % 'mask_check' + "_pts.png", dpi=96)
        plt.close(fig)
        x0, y0, r_grating, border_ring_width, angle = find_mask_params(**pts_dict)
        grating_mask, white_mask, black_mask = utils.create_square_masks(lum_image.shape, x0, y0,
                                                                         r_grating,
                                                                         border_ring_width, angle)
        fig = utils.plot_masked_images(lum_image, [grating_mask, white_mask, black_mask])
        fig.savefig(save_stem % 'mask_check' + "_masks.png")
        plt.close(fig)
        grating_rms = rms_contrast(lum_image[grating_mask])
        df.loc[('grating', 'rms'), 'contrast'] = grating_rms
        # since the masks are boolean, we can simply sum them to get the union
        border_rms = rms_contrast(lum_image[white_mask+black_mask])
        df.loc[('border', 'rms'), 'contrast'] = border_rms
        grating_1d = utils.extract_1d_grating(lum_image, grating_mask, metadata['grating_direction'],
                                              angle)
        border = utils.extract_1d_border(lum_image, white_mask, black_mask, x0, y0, angle)
        _, _, grating_fourier_contrast, grating_fourier_freq = fourier_contrast(grating_1d,
                                                                                plot_flag=False)
        df.loc[('grating', 'fourier'), 'frequency'] = np.abs(grating_fourier_freq)
        df.loc[('grating', 'fourier'), 'contrast'] = grating_fourier_contrast
        _, _, border_fourier_contrast, border_fourier_freq = fourier_contrast(border,
                                                                              plot_flag=False)
        df.loc[('border', 'fourier'), 'frequency'] = np.abs(border_fourier_freq)
        df.loc[('border', 'fourier'), 'contrast'] = border_fourier_contrast
        # we also want the corrected contrast
        if blank_lum_image is not None:
            lum_image_corrected = lum_image / blank_lum_image
            # in some rare cases, there's a 0 in blank_lum_image where there's not a zero in
            # lum_image, which gives an infinity. I'm not sure what to do with that, so throw it to
            # 0
            lum_image_corrected[np.isinf(lum_image_corrected)] = 0
            lum_image_corrected[np.isnan(lum_image_corrected)] = 0
            imageio.imsave(save_stem % 'corrected_luminance' + '_cor_lum.png', lum_image_corrected)
            grating_rms_corrected = rms_contrast(lum_image_corrected[grating_mask])
            border_rms_corrected = rms_contrast(lum_image_corrected[white_mask+black_mask])
            grating_1d = utils.extract_1d_grating(lum_image_corrected, grating_mask,
                                                  metadata['grating_direction'], angle)
            border = utils.extract_1d_border(lum_image_corrected, white_mask, black_mask, x0, y0,
                                             angle)
            _, _, grating_fourier_corrected, _ = fourier_contrast(grating_1d, plot_flag=False)
            _, _, border_fourier_corrected, _ = fourier_contrast(border, plot_flag=False)
            df.loc[('grating', 'rms'), 'lum_corrected_contrast'] = grating_rms_corrected
            df.loc[('border', 'rms'), 'lum_corrected_contrast'] = border_rms_corrected
            df.loc[('grating', 'fourier'), 'lum_corrected_contrast'] = grating_fourier_corrected
            df.loc[('border', 'fourier'), 'lum_corrected_contrast'] = border_fourier_corrected
        grating_rms_freq, border_rms_freq = get_frequency(lum_image, metadata, grating_mask,
                                                          white_mask, black_mask, x0, y0, angle)
        df.loc[('grating', 'rms'), 'frequency'] = np.abs(grating_rms_freq)
        df.loc[('border', 'rms'), 'frequency'] = np.abs(border_rms_freq)
        return df.reset_index(), demosaiced_image, standard_RGB, lum_image


def mtf(fnames, force_run=False, save_path='mtf.csv'):
    """run the entire MTF calculation

    we only re-run this on the images that have not already been analyzed. to over-rule that
    behavior (and run on all specified images), set force_run=True

    fnames: list of strs, the paths to the NEF files to analyze
    """
    if type(fnames) is not list:
        fnames = [fnames]
    # construct the list of tuples (raw_image_filename, preproc_type) to analyze. we make it a list
    # so we can iterate through it multiple times.
    tuples_to_analyze = list(itertools.product(fnames, ['no_demosaic', 'dcraw_vng_demosaic',
                                                        'dcraw_ahd_demosaic']))
    try:
        orig_df = pd.read_csv(save_path)
    except FileNotFoundError:
        orig_df = pd.DataFrame({'filename': [], 'preprocess_type': []})
    if not force_run:
        tmp = []
        for f, preproc in tuples_to_analyze:
            # then this exact pair is not in orig_df and so we should analyze it. we use isin
            # instead of == because isin will work for the empty dataframe as well
            if orig_df[(orig_df.filename.isin([f])) &
                       (orig_df.preprocess_type.isin([preproc]))].empty:
                tmp.append((f, preproc))
        tuples_to_analyze = tmp
    else:
        # iterate through the tuples to analyze and drop all of them from orig_df
        for f, preproc in tuples_to_analyze:
            idx = orig_df[(orig_df.filename.isin([f])) &
                          (orig_df.preprocess_type.isin([preproc]))].index
            orig_df.drop(idx)
    if tuples_to_analyze:
        print("Analyzing:\n\t%s" % "\n\t".join([", ".join(tup) for tup in tuples_to_analyze]))
    else:
        print("No new images to analyze, exiting...")
        return
    dfs = []
    blank_lum_imgs = {}
    for f, preproc in tuples_to_analyze:
        f_stem = os.path.splitext(os.path.split(f)[-1])[0]
        blank_lum_name = utils.find_corresponding_blank(f_stem)
        if (blank_lum_name, preproc) not in blank_lum_imgs.keys():
            blank_fullname = os.path.join(os.path.split(f)[0], blank_lum_name) + '.NEF'
            blank_lum_imgs[(blank_lum_name, preproc)], _, _, _, _ = create_luminance_image(blank_fullname,
                                                                                           preproc)
        if (f_stem, preproc) in blank_lum_imgs.keys():
            print("%s is a blank image, skipping..." % f)
            continue
        print("Analyzing %s with preproc method %s" % (f, preproc))
        blank = blank_lum_imgs[(blank_lum_name, preproc)]
        df, demosaic, std, lum = main(f, preproc, blank)
        dfs.append(df)
    # if this function is only called on a blank image, there will be nothing in the dfs list to
    # concatenate
    if len(dfs) > 0:
        df = pd.concat(dfs)
        if orig_df is not None:
            df = pd.concat([orig_df, df])
        df = df.reset_index(drop=True)
        df['image_cycles'] = df.image_content.apply(lambda x: int(x.replace(' cyc/image', '')))
        df['display_freq'] = df.apply(lambda x: x.image_cycles / int(x.grating_size.replace(' pix', '')), 1)
        df.to_csv(save_path, index=False)
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate the MTF using the specified images")
    parser.add_argument("images", nargs='+',
                        help=("What images to calculate contrast for and use to create the MTF"))
    parser.add_argument("--force_run", "-f", action="store_true",
                        help=("Whether to run on all specified images or not. If not passed, we "
                              "skip all images that have already been analyzed"))
    parser.add_argument("--save_path", "-s", default='mtf.csv',
                        help="Path to save the output dataframe (as a csv)")
    args = vars(parser.parse_args())
    mtf(args['images'], args['force_run'], args['save_path'])
