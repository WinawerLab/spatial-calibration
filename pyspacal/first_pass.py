#!/usr/bin/python
"""functions to quickly summarize mean and std dev of sensor values
"""

import argparse
import itertools
import numpy as np
import pandas as pd
from . import DataFrameArray as dfa
from . import mtf
from . import utils
from . import camera_data


def preprocess(fname, preproc_method):
    """get the preprocessed image with its grating mask

    this loads in the appropriate preprocessed image, demosaics it and averages across RGB sensors,
    and finds the grating mask

    returns: grating (the masked 2d image), cyc_len_pix, metadata
    """
    img, metadata = utils.load_img_with_metadata(fname, preproc_method)
    if img.ndim == 2:
        # then it hasn't been demosaiced and we do it manuallly
        mask, kernels = mtf.create_mosaic_mask(img.shape,
                                               camera_data.BAYER_MATRICES[metadata['camera']])
        raw_demosaiced_image = mtf.demosaic_image(img, mask)
        demosaiced_image = mtf.block_avg(raw_demosaiced_image)
    else:
        # then it has and we just have to rearrange the axes so that RGB is one the first one
        demosaiced_image = img.transpose((2, 0, 1))
    # we just average across the RGB dimension
    demosaiced_image = demosaiced_image.mean(0)
    pts_dict = utils.load_pts_dict(metadata['filename'], demosaiced_image.shape)
    x0, y0, r_grating, border_ring_width, angle = mtf.find_mask_params(**pts_dict)
    grating_mask, white_mask, black_mask = utils.create_square_masks(demosaiced_image.shape,
                                                                     x0, y0, r_grating,
                                                                     border_ring_width, angle)
    grating_1d = utils.extract_1d_grating(demosaiced_image, grating_mask,
                                          metadata['grating_direction'], angle)
    n_cycs = int(metadata['image_content'].split(' ')[0])
    # This is the size of each cycle in pixels
    cyc_len_pix = len(grating_1d) / n_cycs
    grating = demosaiced_image * grating_mask
    grating[grating == 0] = np.nan
    return grating, cyc_len_pix, metadata


def calc_mean_and_stddev(grating, box_size, step_size, buffer_pix=(0, 0)):
    """calculate the means and stddev of the grating and return it

    buffer_pix: 2-tuple of ints. how many extra pixels to avoid near the edge
    """
    indices = np.where(~np.isnan(grating))
    indices = np.vstack([np.min(indices, 1)+20, np.max(indices, 1)])
    means = []
    stddevs = []

    for x in range(*indices[0], step_size):
        if x+box_size+buffer_pix[0] > np.max(indices[0]):
            continue
        means.append([])
        stddevs.append([])
        for y in range(*indices[1], step_size):
            if y+box_size+buffer_pix[1] > np.max(indices[1]):
                continue
            tmp = grating[x:x+box_size, y:y+box_size]
            mn = np.nanmean(tmp)
            sd = np.sqrt(np.nanmean(np.square(tmp - mn)))
            means[-1].append(mn)
            stddevs[-1].append(sd)
    means = np.array(means)
    stddevs = np.array(stddevs)
    return means, stddevs


def main(fnames, save_path='summarize.csv', box_size_multiple=1, step_size_multiple=1,
         buffer_pix=(0, 0)):
    """summarize mean and std dev of sensor values

    box_size_multiple and step_size_multiple are both multiplied by the length of each cycle in
    pixels in order to get the size of the box and step, respectively. they can both be less than
    1, but we need to cast them as ints

    """
    if type(fnames) is not list:
        fnames = [fnames]
    # construct the list of tuples (raw_image_filename, preproc_type) to analyze. we make it a list
    # so we can iterate through it multiple times.
    tuples_to_analyze = list(itertools.product(fnames, ['no_demosaic', 'dcraw_vng_demosaic',
                                                        'dcraw_ahd_demosaic']))
    df = []
    for f, preproc in tuples_to_analyze:
        grating, cyc_len_pix, metadata = preprocess(f, preproc)

        box_size = int(cyc_len_pix * box_size_multiple)
        step_size = int(cyc_len_pix * step_size_multiple)
        if box_size == 0:
            raise Exception("with box_size_multiple %s and cycle length %s pixels, box_size was 0"
                            " pixels!" % (box_size_multiple, cyc_len_pix))
        if step_size == 0:
            raise Exception("with step_size_multiple %s and cycle length %s pixels, step_size was "
                            "0 pixels!" % (step_size_multiple, cyc_len_pix))
        means, stddevs = calc_mean_and_stddev(grating, box_size, step_size, buffer_pix)

        data = metadata.copy()
        data.update({'means': means, 'stddevs': stddevs, 'box_size': box_size,
                     'step_size': step_size, 'cycle_size_pix': cyc_len_pix,
                     'box_size_multiple': box_size_multiple,
                     'step_size_multiple': step_size_multiple})
        df.append(dfa.DataFrameArray(data))
    df = pd.concat(df).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarize mean and std dev of sensor values of"
                                     " specified images")
    parser.add_argument("images", nargs='+',
                        help=("What images to calculate contrast for and use to create the MTF"))
    parser.add_argument("--save_path", "-p", default='summarize.csv',
                        help="Path to save the output dataframe (as a csv)")
    parser.add_argument("--box_size_multiple", "-b", type=float, default=1,
                        help="The box we use to average is cyc_len_pix * box_size_multiple")
    parser.add_argument("--step_size_multiple", "-s", type=float, default=1,
                        help=("The step size we use as we move the averaging box around is "
                              "cyc_len_pix * step_size_multiple"))
    args = vars(parser.parse_args())
    main(args['images'], args['save_path'], args['box_size_multiple'], args['step_size_multiple'])
