# spatial-calibration

Python code to check the calibration of displays

This will involve displaying a bunch of images on your display, taking
pictures of them with a high-quality camera (turning all the bells and
whistles off, ideally storing as RAW images), and then running them
through the included analysis to determine the displayed contrast.

# Requirements

All python requirements are handled via the `environment.yml` file,
but you will also
need [dcraw](https://www.cybercom.net/~dcoffin/dcraw/)
and [exiftool](https://www.sno.phy.queensu.ca/~phil/exiftool/)
installed (both are available in the default Ubuntu repositories and
look easy to install on OSX as well).

The folder `software_versions` contains logs showing the versions used
of `dcraw`, `exiftool`, `gphoto2`, and the python packages, as well as
a simple python script used to ease the collection of images with
`gphoto2`. Unless otherwise specified in the `environment.yml` file,
it is not thought that the specific version matters for the gathering
of the data, running of the analysis, or the analysis results.

All analyses only run on Linux (Ubuntu 18.04 LTS and CentOS Linux 7),
but they should work with little to no tweaking on Mac. No guarantee
they will work on any Microsoft OS.

If you are unfamiliar with setting up python environments, install
[miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS, open
your terminal, navigate to wherever you have downloaded this repo, and run the
following:

``` sh
conda env -f environment.yml
```

This will create a virtual environment with the necessary python libraries. Once
you activate the environment by running `conda activate calibration`, `python`
will use the version installed in that environment and you'll have access to all
the libraries.

There are two main ways of getting jupyter working so you can view the included
notebooks:

1. Install jupyter in this calibration environment: 

``` sh
conda activate calibration
conda install -c conda-forge jupyterlab
```

   This is easy but, if you have multiple conda environments and want to use
   Jupyter notebooks in each of them, it will take up a lot of space.
   
2. Use [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels):

``` sh
# activate your 'base' environment, the default one created by miniconda
conda activate 
# install jupyter lab and nb_conda_kernels in your base environment
conda install -c conda-forge jupyterlab
conda install nb_conda_kernels
# install ipykernel in the calibration environment
conda install -n calibration ipykernel
```

   This is a bit more complicated, but means you only have one installation of
   jupyter lab on your machine.
   
In either case, to open the notebooks, navigate to this directory on your
terminal and activate the environment you install jupyter into (`calibration`
for 1, `base` for 2), then run `jupyter` and open up the notebook. If you
followed the second method, you should be prompted to select your kernel the
first time you open a notebook: select the one named "calibration".

# Data

If you wish to re-run the analysis using pictures taken of the projector in
NYU's Prisma during summer 2018 (as used by the [spatial frequency preferences
project](https://osf.io/afs85/)), then you can get the raw data from the
[OSF](https://osf.io/vmu3s/) (`prisma_raw_images.tar.gz`). The OSF also contains
`mtf-spline.svg` and `mtf-spline.pkl`, which give the relationship between the
modulation transfer function and the display frequency, as described
[below](#use), and `mtf.csv`, which contains the data used to generate those
files, as well as the MTF that results from different ways of measuring the
contrast, preprocessing the RAW images, and photos taken at different times.

# Use

If you have set up the environment, as described [above](#requirements), and will
be using `snakemake`, as described below, this will be handled automatically.

Open up `Snakefile` and edit the `DATA_DIR` variable to wherever you would like
to store the data (by default, this is the `data/` directory within this
directory). Then, navigate to this directory in your terminal and run:

``` sh
conda activate calibration
snakemake -n -prk mtf_spline
```

Assuming everything is correctly configured, `snakemake` should wait a while it
builds the analysis DAG, then it should print out the many steps necessary to
create the final output (see the [overview](#overview) section and the included
Jupyter notebook `MTF.ipynb` to understand what steps are taken). To actually
run the command, remove the `-n` flag (and you may want to add `-j N`, where `N`
is an integer, in order to parallelize the jobs when possible).

At the end of this process, `DATA_DIR` will contain `mtf-spline.svg` and
`mtf-spline.pkl`. Open up the svg file to view the modulation transfer function
as a function of the display frequency in cycles per pixel (on a log scale).
`mtf-spline.pkl` is a pickled python function (an interpolation of the data, as
shown by the orange line in the plot), which takes spatial frequencies and
returns the MTF. To load it in, run:

``` python
import pickle
with open('path/to/mtf-spline.pkl', 'rb') as f:
    mtf = pickle.load(f)
```

A simple example use:

``` python
# some spatial frequencies
sfs = np.logspace(-8, -1, base=2)
mtf(sfs)
```

See `stimuli.py` in [the spatial frequency
repo](https://github.com/billbrod/spatial-frequency-preferences/blob/master/sfp/stimuli.py#L1104)
for a more involved example of how it can be used.

# Overview

The goal of this project is to estimate the modulation transfer
function (MTF) of a display, in order to examine how much contrast is
loss at higher spatial frequencies. This happens because no lens (for
a projector) or other display is perfect; all of them have a
pointspread function of some form, which serves to blur (i.e., run a
lowpass filter on) the displayed image, thus reducing the contrast at
the higher frequencies. We want to estimate the approximate shape of
this contrast vs. spatial frequency curve. As vision scientists, we
will then use this information to construct stimuli that invert the
MTF (reducing the amplitude of the low frequencies so that, when
displayed, all frequencies have the same contrast), in order to
evaluate how large of an effect this loss of contrast is.

In order to estimate the modulation transfer function, we created a
series of square-wave gratings in a square aperture, each with a total
width of 256 pixels, running from 1 cycle per image (256 pixels per
cycle) up to the Nyquist limit, 128 cycles per image (2 pixels per
cycle). We do this for both vertical and horizontal gratings. Each of
these is surrounded by a white border and a black border, each with a
width of 128 pixels, so that our overall displayed image fills a 512
by 512 pixel square in the center. These are displayed using PsychoPy,
which enables us to perform luminance correction (so that a pixel
whose value is 255 is twice as bright as one whose value is 127). We
then take pictures of these images using a DSLR camera (we used a
Nikon D90, but any DSLR should work; we used a standard consumer
lens). The ISO was set to 200, the lowest available (in order to
reduce sensor noise); the f-number (aperture size) was set to f16
(this was chosen so that the f-number was reasonably high / the
aperture was reasonably small, in order to avoid lens artifacts, but
not too high / too small, in order to avoid diffraction artifacts);
the shutter speed was then set to ensure a reasonable amount of light
reached the sensor (i.e., that the max recorded sensor value was about
80% of the max possible), while being long enough to avoid temporal
aliasing from the refresh rate of the display and short enough to
avoid shaking from small vibrations (the camera was set up on a
linoleum floor on the ground floor of a building in New York
University, directly above some subway tunnels). The camera was set up
on a tripod, placed as close to the display as possible (to avoid any
zoom) so that the full 512 pixels of the displayed image filled the
picture, and connected via USB cord to a laptop running Ubuntu 18.04
(this tethering enabled us to capture images using
[gphoto2](http://www.gphoto.org/) (installed from Ubuntu package
directory) without touching the camera, minimizing differences from
picture-to-picture). The camera was focused manually as best as
possible and photos were taken at two different focus settings to
ensure they were approximately correct. The pictures were in raw image
format (Nikon's variant is NEF) in order to avoid any post-processing
performed by the camera (which could, for example, attempt to sharpen
the image).

After the photos were taken, they were transferred to a laptop (the
one running Ubuntu 18.04 mentioned above) and processed with the code
found in this repository on some combination of the laptop, the NYU
High Performance Computing cluster, and the Linux machines found in
the NYU Lab for Computational Vision. There are two separate analyses,
"first pass" and "mtf". Both of them start with demosaicing the raw
image, which converts the images from raw sensory values into RGB or
greyscale images. In order to make sure that this didn't have a large
effect on the outcome, three different demosaicing methods were tried:
a naive block-average method (average together the camera sensors in a
2x2 block so that you end up with a down-sampled greyscale image),
dcraw's adaptive homogeneity-directed (AHD) interpolation, and dcraw's
variable number of gradients (VNG) interpolation. This did not have
any effect on the analysis outcome. An important step of both analyses
is also the extraction of the grating found in the center of the
image. Instead of relying on computer vision methods, such as
edge-detection, we do this by simply finding the locations of the
edges of the grating, the white border, and the black border (in
pixels), recording them in `pyspacal/camera_data.py`, and using
`scipy.optimize.least_squares` to simultaneously find the center of
the square, the width of the grating, and the width of the border
(both white and black borders have the same width) that best satisfies
these points. The resulting square is used to construct masks
corresponding to the grating and the two borders.

First pass:
1. If the image is not already greyscale, average across the RGB
   dimension in order to convert it to greyscale.
2. Extract the grating, as described above.
3. Move a box across the grating, calculating the mean and standard
   deviation of the pixel values within the box. The box and step size
   should be either the same size as the grating's period or an
   integer multiple of it.
4. Display these images and investigate. The mean image allows us to
   investigate how the average luminance changes, while the standard
   deviation image allows us to investigate how the contrast
   changes. Both can be viewed as a function of space (i.e., how they
   change within an image) and as a function of spatial frequency
   (i.e., how they change across images).
5. We see that there is a hot-spot (the mean luminance is not constant
   across the image) and that the standard deviation decreases as
   spatial frequency increases. Mean luminance does not change as a
   function of spatial frequency, and the standard deviation has the
   same pattern of difference across an image (since the contrast is
   proportional to the mean divided by the standard deviation, the
   contrast does not change as a function of space)
   
MTF:
1. We either convert the image directly to greyscale (averaging across
   the RGB dimension), or we first convert it to a luminance image,
   following the procedure in Tkacik et al, 2011:

  - Standardize the RGB values. Tkacik et al show their camera sensor
    values are linear with respect to shutter speed, aperture, and
    ISO, and we assume this; we do not check this.
  - Convert the RGB values to LMS (cones), using the sensitivities of
    the Nikon D90 camera from the [camspec
    database](www.gujinwei.org/research/camspec/db.html) (which give
    the responses of the R, G, and B sensors as a function of
    wavelength) and the [2 degree cone
    fundamentals](www.cvrl.org/cones.htm) (which give the responses of
    the L, M, and S cones as a function of wavelength).
  - Convert LMS to luminance using the values given in the Tkacik et
    al paper.
	
2. Divide the greyscale or luminance image by the greyscale or
   luminance version of an image taken of a mid-grey stimulus on the
   same display, taken during the same session with the same
   settings. This will effectively correct for the hot-spot problem
   mentioned above, leaving only the variation in contrast as a
   function of spatial frequency.
3. Extract the grating, as described above.
4. Calculate the contrast of the grating, $I$, using one of three
   contrast measures:

   - Michelson contrast: $\frac{\max(I) - \min(I)}{\max(I) + \min(I)}$
   - Root mean squared contrast: $\frac{\sqrt{\frac{\sum{I_i -
     \bar{I}}}{n}}}{\bar{I}}$, where $I_i$ is the value of pixel $i$
     within the the grating $I$.
   - Fourier contrast: convert the 2d grating into a 1d grating by
     averaging along either the vertical or horizontal direction
     (depending on whether the grating was horizontal or vertical),
     and take the amplitude of the square wave's fundamental (we do
     not calculate the frequency of the fundamental, we simply take
     the Fourier transform of the 1d grating and find the frequency
     with the max amplitude).
 
5. Plot the contrast of each grating as a function of its frequency
   (which we know based on how we constructed the grating).
6. This plot shows the modulation transfer function of the
   display. For a given image, Michelson > RMS > Fourier. Because the
   display's pointspread function acts as a lowpass filter or
   blurring, the Michelson contrast will be least affected (the blur
   will have the greatest effect on the pixels near the transition of
   the square wave grating and will need to grow fairly large before
   affecting the extreme values). The RMS contrast is more affected by
   the blur than the Michelson contrast, but is affected by multiple
   frequencies, whereas the Fourier contrast comes from the
   fundamental and thus will be the lowest. The shape of the MTF curve
   is the same for all three contrast measures, but the actual
   contrast values differ. The use of luminance image or the greyscale
   image calculated directly from sensor values does not seem to have
   any effect on the measured MTF.

- Ga\vsper Tka\vcik, Garrigan, P., Ratliff, C., Grega Mil\vcinski,
  Klein, J. M., Seyfarth, L. H., Sterling, P., ... (2011). Natural
  images from the birthplace of the human eye. PLoS} {ONE, 6(6),
  20409. http://dx.doi.org/10.1371/journal.pone.0020409

