#!/usr/bin/python
"""utility functions
"""

import matplotlib
# the PyQt5 backend causes an issue with the environment necessary for psychopy, for some
# reason. we explicitly tell it to use the SVG backend to avoid that. We set warn=False because the
# notebook uses a different backend and will spout out a big warning to that effect; that's
# unnecessarily alarming, so we hide it.
matplotlib.use('SVG', warn=False)
import os
import glob
import shutil
import subprocess
import imageio
import exifread
import numpy as np
import pandas as pd
import seaborn as sns
from . import camera_data
import matplotlib.pyplot as plt


def show_im_well(img, ppi=96, zoom=1):
    ppi = float(ppi)
    fig = plt.figure(figsize=(zoom*img.shape[1]/ppi, zoom*img.shape[0]/ppi), dpi=ppi)
    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig_width, fig_height = bbox.width*fig.dpi, bbox.height*fig.dpi
    ax= fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.imshow(img, cmap='gray', interpolation='none')
    return fig


def _get_preprocessed_fname(raw_fname, preprocess_type):
    """returns the preprocessed fname without an extension
    """
    preprocessed_fname = raw_fname.replace('raw', os.path.join("preprocessed",  preprocess_type))
    return os.path.splitext(preprocessed_fname)[0]
    


def preprocess_image(raw_fname, preprocess_type):
    """check if the appropriately preprocessed image has been created and, if not, try to create it
    """
    try:
        _ = find_preprocessed_file(raw_fname, preprocess_type)
    except Exception:
        preprocessed_fname = _get_preprocessed_fname(raw_fname, preprocess_type)
        if preprocess_type == 'no_demosaic':
            subprocess.call(['dcraw', '-4', '-d', raw_fname])
            shutil.move(os.path.splitext(raw_fname)[0] + ".pgm", preprocessed_fname + ".pgm")
        elif preprocess_type == 'dcraw_vng_demosaic':
            subprocess.call(['dcraw', '-4', '-q', '1', '-T', raw_fname])
            shutil.move(os.path.splitext(raw_fname)[0] + ".tiff", preprocessed_fname + ".tiff")
        elif preprocess_type == 'dcraw_ahd_demosaic':
            subprocess.call(['dcraw', '-4', '-q', '3', '-T', raw_fname])
            shutil.move(os.path.splitext(raw_fname)[0] + ".tiff", preprocessed_fname + ".tiff")
        elif preprocess_type == 'nikon_demosaic':
            raise Exception("Cannot find nikon_demosaic preprocessed image and cannot create it"
                            " myself; use Nikon's Capture NX-D software to do this yourself")
        else:
            raise Exception("Unsure how to preprocess image for preprocess_type %s" % preprocess_type)


def find_preprocessed_file(raw_fname, preprocess_type):
    """find, load, and return preprocessed file from the raw file
    """
    preprocessed_fname = _get_preprocessed_fname(raw_fname, preprocess_type) + ".*"
    preprocessed_file = glob.glob(preprocessed_fname)
    if len(preprocessed_file) != 1:
        raise Exception("Can't find unique preprocessed file! %s" % preprocessed_file)
    return imageio.imread(preprocessed_file[0]).astype(np.double)


def load_img_with_metadata(raw_fname, preprocess_type):
    """load image and grab relevant metadata

    note that fname should be the path to the raw image (contained within the data/raw directory)

    preprocess_type should be the name of the directory under preprocessed/ where the preprocessed
    image can be found
    """
    assert "raw" in raw_fname, "File should be in data/raw/ directory!"
    with open(raw_fname, 'rb') as f:
        tags = exifread.process_file(f)
    # The f number and exposure time are Ratio and we want them as floats
    metadata = {'f_number': tags['EXIF FNumber'].values[0].num / float(tags['EXIF FNumber'].values[0].den),
                'iso': tags['EXIF ISOSpeedRatings'].values[0],
                'exposure_time': tags['EXIF ExposureTime'].values[0].num / float(tags['EXIF ExposureTime'].values[0].den),
                'focus_mode': tags['MakerNote FocusMode'].values.strip(),
                'filename': os.path.splitext(os.path.split(raw_fname)[-1])[0],
                'camera': tags['Image Model'].values,
                'preprocess_type': preprocess_type}
    metadata['context'] = camera_data.IMG_INFO[metadata['filename']][0]
    metadata['content'] = camera_data.IMG_INFO[metadata['filename']][1]
    img = find_preprocessed_file(raw_fname, preprocess_type)
    return img, metadata


def load_cone_fundamental_matrix(fname='data/linss2_10e_5.csv', min_wavelength=400,
                                 max_wavelength=720, target_wavelength_incr=10):
    """load in the cone fundamental matrix and restrict its values

    we load in the cone fundamental matrix at the specified location (download it from
    http://www.cvrl.org/cones.htm) and then restrict it to the appropriate domain and sampling
    amount. this should match what you have for the camera RGB sensitivities. the default values
    are for the sensitivities from the camspec database. if you're using the sensitivities from
    Tkacik et al, max_wavelength should be 700.

    target_wavelength_incr: int, the increment (in nanometers) that we want to sample the cone
    sensitivities by. The cone fundamentals generally have a finer sampling than the camera
    sensitivities do, so we need to sub-sample it. this must be an integer multiple of the cone
    fundamental's sampling.
    """
    s_lms = pd.read_csv('data/linss2_10e_5.csv', header=None,
                        names=['wavelength', 'l_sens', 'm_sens', 's_sens'])
    s_lms = s_lms.fillna(0)
    s_lms = s_lms[(s_lms.wavelength>=min_wavelength) & (s_lms.wavelength<=max_wavelength)]
    data_wavelength_incr = np.unique(s_lms['wavelength'][1:].values - s_lms['wavelength'][:-1].values)
    if len(data_wavelength_incr) > 1:
        raise Exception("The cone fundamental matrix found at %s does not evenly sample the "
                        "wavelengths and I'm unsure how to handle that!" % fname)
    subsample_amt = target_wavelength_incr / data_wavelength_incr[0]
    if subsample_amt < 1:
        raise Exception("You want me to sample the wavelengths at a finer resolution than your "
                        "data provides! data: %s, target: %s" % (data_wavelength_incr[0],
                                                                 target_wavelength_incr))
    if int(subsample_amt) != subsample_amt:
        raise Exception("Can't sample wavelengths every %s nm as desired because data samples "
                        "every %s nm. The target must be an integer multiple of the data!" %
                        (target_wavelength_incr, data_wavelength_incr[0]))
    s_lms = s_lms[::int(subsample_amt)].reset_index(drop=True)
    s_lms_mat = s_lms[['l_sens', 'm_sens', 's_sens']].values    
    return s_lms_mat


def load_camspec_sensitivity_matrix(fname='data/camspec_database.txt', camera=None):
    """load in the camera sensitivities from the camspec database

    Because of how `camspec_database.txt` is formatted, this is slightly complicated. The file is
    many lines long, with the following pattern:

    ```
    camera_name_1
    r_sens_400   r_sens_410   ...   r_sens_720
    g_sens_400   g_sens_410   ...   g_sens_720
    b_sens_400   b_sens_410   ...   b_sens_720
    camera_name_2
    ...
    ```

    where those three spaces in the sensitivity lines represent a tab. We therefore read in the
    file line by line, expecting the above pattern, reading the camera names and the three
    sensitivities into different arrays. We also construct the wavelengths array, using the values
    shown in the plot on the website. We then tile the wavelengths and repeat the camera names so
    that all arrays are the same length, which allows us to create a nice Dataframe with all this
    data. Finally, we (optionally) select only the camera we're interested in and convert the
    sensitivites to a matrix.

    if camera is None, we return the dataframe containing all the values. if camera is not None, we
    select the corresponding data and convert the sensitivities to a matrix
    """
    with open(fname) as f:
        tmp = f.readlines()

    # we know (from the website) that the sensitivities correspond to 400 through 720 nm,
    # inclusive.
    wavelengths = np.arange(400, 721, 10)
    n_wavelengths = len(wavelengths)
    # This file has the name of the camera on the 1st line, then 3 lines detailing the R, G, and B
    # sensitivities, before repeating with the next camera therefore, every 4th line contains the
    # name of the camera, the others are sensitivities
    cameras = []
    r_sens = []
    g_sens = []
    b_sens = []
    for i, t in enumerate(tmp):
        t = t.strip().split('\t')
        if i % 4 == 0:
            cameras.append(t)
        elif i % 4 == 1:
            r_sens.append([float(j) for j in t])
        elif i % 4 == 2:
            g_sens.append([float(j) for j in t])
        elif i % 4 == 3:
            b_sens.append([float(j) for j in t])        

    # in order to easily parse this as a dataframe, we tile / repeat the wavelengths / cameras
    # (respectively) so that those arrays are the same length as the sensitivity ones
    wavelengths = np.tile(wavelengths, len(cameras))
    cameras = np.repeat(np.array(cameras).flatten(), n_wavelengths)
    r_sens = np.array(r_sens).flatten()
    g_sens = np.array(g_sens).flatten()    
    b_sens = np.array(b_sens).flatten()    

    s_rgb = pd.DataFrame({'camera': cameras, 'r_sens': r_sens, 'g_sens': g_sens, 'b_sens': b_sens,
                          'wavelength': wavelengths})
    if camera is not None:
        # to make matching easier, we downcase both the target and all the values
        s_rgb.camera = s_rgb.camera.apply(lambda x: x.lower())
        s_rgb = s_rgb[s_rgb.camera==camera.lower()]
        s_rgb = s_rgb[['r_sens', 'g_sens', 'b_sens']].values    

    return s_rgb


def load_pts_dict(filename, img_shape):
    """load in the pts dict corresponding to this image and get it in the right format

    the values in the PTS_DICT constant are those that you can find using photoshop / inkscape. in
    order to convert them to the correct numpy array indices, we need to go from (i, j) -> (i,
    img.shape[0]-j)

    you should call `check_pts_dict` after you load this in to visually check that everything looks
    right.
    """
    pts_dict_tmp = camera_data.PTS_DICT[filename].copy()
    pts_shape = pts_dict_tmp.pop('image_size')
    # if ((not (img_shape[0] / pts_shape[0]).is_integer()) or
    #     (not (img_shape[1] / pts_shape[1]).is_integer())):
    #     if ((not (pts_shape[0] / img_shape[0]).is_integer()) or
    #         (not (pts_shape[1] / img_shape[1]).is_integer())):
    #         raise Exception("The points must be defined on an image that is integer down- or up-"
    #                         "sampled from the image you're trying to extract the grating from! "
    #                         "img_shape: (%s, %s), pts_shape: (%s, %s)" %
    #                         (img_shape[0], img_shape[1], pts_shape[0], pts_shape[1]))
    img_rescale = (img_shape[0] / pts_shape[0], img_shape[1] / pts_shape[1])
    pts_dict = {}
    for k, v in pts_dict_tmp.items():
        if type(v) is not list:
            pts_dict[k] = (int(img_rescale[0] * v[0]), int(img_shape[0] - img_rescale[1] * v[1]))
        else:
            pts_dict[k] = [(int(img_rescale[0] * i), int(img_shape[0] - img_rescale[1] * j)) for i, j in v]
    return pts_dict


def _plot_pts_on_img(img, pts, zoom=1):
    fig=show_im_well(img, zoom=zoom)
    for p in pts:
        plt.scatter(p[1], p[0])    
    return fig


def check_pts_dict(img, pts_dict, zoom=1):
    # this is all just to get these pts into the arrangement that _plot_pts_on_img expects
    pts = [np.array(v).T[[1, 0]] for v in pts_dict.values()]
    return _plot_pts_on_img(img, pts, zoom)


def create_circle_masks(img_shape, circle_ctr_x, circle_ctr_y, grating_radius, border_ring_width):
    """create circle masks to extract relevant regions

    this uses the outputs from mtf.find_mask_params and returns three boolean masks to extract: the
    grating, the white region of the border, and the black region of the border
    """
    xgrid, ygrid = np.meshgrid(range(img_shape[1]), range(img_shape[0]))
    grating_mask = np.zeros(img_shape)
    grating_mask[(xgrid - circle_ctr_x)**2 + (ygrid - circle_ctr_y)**2 <= grating_radius**2] = 1
    white_mask = np.zeros(img_shape)
    white_mask[(xgrid - circle_ctr_x)**2 + (ygrid - circle_ctr_y)**2 <= (grating_radius+border_ring_width)**2] = 1
    white_mask -= grating_mask
    black_mask = np.zeros(img_shape)
    black_mask[(xgrid - circle_ctr_x)**2 + (ygrid - circle_ctr_y)**2 <= (grating_radius+2*border_ring_width)**2] = 1
    black_mask -= (grating_mask + white_mask)
    return grating_mask.astype(bool), white_mask.astype(bool), black_mask.astype(bool)


def create_circle_outlines(img_shape, circle_ctr_x, circle_ctr_y, grating_radius,
                           border_ring_width, edge_tol=10):
    """create outlines of the circular masks used to extract relevant regions

    this does basically the same thing as `create_circle_masks` but returns a list of indices into
    the image, so they can be plotted on the image to double-check that things look good (use
    `check_circle_outlines` to do so).

    the higher the value of edge_tol, the more points we'll find.
    """
    xgrid, ygrid = np.meshgrid(range(img_shape[1]), range(img_shape[0]))
    grating_mask = np.zeros(img_shape)
    grating_mask[np.abs((xgrid - circle_ctr_x)**2 + (ygrid - circle_ctr_y)**2 - grating_radius**2) < edge_tol] = 1
    white_mask = np.zeros(img_shape)
    white_mask[np.abs((xgrid - circle_ctr_x)**2 + (ygrid - circle_ctr_y)**2 - (grating_radius+border_ring_width)**2) < edge_tol] = 1
    black_mask = np.zeros(img_shape)
    black_mask[np.abs((xgrid - circle_ctr_x)**2 + (ygrid - circle_ctr_y)**2 - (grating_radius+2*border_ring_width)**2) < edge_tol] = 1
    return np.where(grating_mask), np.where(white_mask), np.where(black_mask)


def check_circle_outlines(img, grating_mask_pts, white_mask_pts, black_mask_pts, zoom=1):
    pts = [np.array(p) for p in [grating_mask_pts, white_mask_pts, black_mask_pts]]
    return _plot_pts_on_img(img, pts, zoom)


def plot_masked_images(img, masks):
    fig, axes = plt.subplots(1, len(masks), figsize=(len(masks)*10, 10))
    for ax, mask in zip(axes, masks):
        tmp = img*mask
        tmp[tmp==0] = np.nan
        ax.imshow(tmp, cmap='gray', interpolation='none')
        ax.set(xticks=[], yticks=[])    
    return fig


def plot_masked_distributions(img, masks):
    fig, axes = plt.subplots(1, len(masks), figsize=(len(masks)*5, 5))
    for ax, mask in zip(axes, masks):
        sns.distplot(img[mask], ax=ax)
        ax.set_xlim(0, img.max())
    return fig


def extract_1d_border(img, white_mask, black_mask, x0, y0):
    """
    """
    white_ring = (img*white_mask)[int(y0), :int(x0)]
    black_ring = (img*black_mask)[int(y0), :int(x0)]
    ring = white_ring + black_ring
    ring[ring==0] = np.nan
    ring = ring[~np.isnan(ring)]
    return ring


def extract_1d_grating(img, grating_mask):
    grating = img * grating_mask
    grating[grating==0] = np.nan
    grating_1d = np.nanmean(grating, 0)
    grating_1d = grating_1d[~np.isnan(grating_1d)]
    return grating_1d
