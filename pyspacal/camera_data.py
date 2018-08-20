"""some camera data that we need to store somewhere
"""

import numpy as np

# We "pythonify" the conventions in the Tkačik et al paper, so R=0, G=1, B=2
BAYER_MATRICES = {
    "NIKON D90": np.array([[1, 2], [0, 1]]),
    "NIKON D70": np.array([[2, 1], [1, 0]]),
    }


IMG_INFO = {
    'DSC_0080': ['monitor', '1 cyc/image'],
    'DSC_0081': ['monitor', '2 cyc/image'],
    'DSC_0082': ['monitor', '4 cyc/image'],
    'DSC_0083': ['monitor', '8 cyc/image'],
    'DSC_0084': ['monitor', '16 cyc/image'],
    'DSC_0085': ['monitor', '32 cyc/image'],
    'DSC_0086': ['monitor', '64 cyc/image'],
    'DSC_0087': ['monitor', '128 cyc/image'],
    'DSC_0088': ['monitor', 'blank'],
    }

# the key specifies which image we're referring to and should be the filename as saved in the
# metadata dictionary (that is, the filename with no directory or extension). each value here is a
# dictionary defining the edges of the various circles for the different images. they must
# therefore have the following keys: 'square_ctr', 'grating_edge', 'white_edge', 'black_edge', and
# 'image_size'. The two numbers in each tuple are the pixels, as you can find using inkscape,
# etc. THIS SHOULD NOT BE USED DIRECTLY. Use the function `load_pts_dict`, because (in order to get
# the values here to correspond to the right image indices), we need to subtract the second number
# in each tuple from img.shape[0]

# annoyingly, image_size has the opposite order of numbers that the points have, because image_size
# must be compared to image.size (which returns the numbers as y, x) while the points should have
# the scatter-like order x, y -- I SHOULD PROBABLY CHANGE THIS
PTS_DICT = {"DSC_0080":
            {"square_ctr": (2148, 1456),
             "grating_edge": [(2071, 2149), (1441, 1593), (1856, 732), (2644, 2143), (2854, 1290),
                              (2586, 731), (1913, 736)],
             "white_edge": [(2200, 2506), (3206, 1347), (2145, 376), (1073, 1425)],
             "black_edge": [(3566, 1368), (709, 1539), (725, 2563), (3571, 66)],
             "image_size": (2868, 4310)},
            "DSC_0081":
            {"square_ctr": (2148, 1456),
             "grating_edge": [(2071, 2149), (1441, 1593), (1856, 732), (2644, 2143), (2854, 1290),
                              (2586, 731), (1913, 736)],
             "white_edge": [(2200, 2506), (3206, 1347), (2145, 376), (1073, 1425)],
             "black_edge": [(3566, 1368), (709, 1539), (725, 2563), (3571, 66)],
             "image_size": (2868, 4310)},
            "DSC_0082":
            {"square_ctr": (2148, 1456),
             "grating_edge": [(2071, 2149), (1441, 1593), (1856, 732), (2644, 2143), (2854, 1290),
                              (2586, 731), (1913, 736)],
             "white_edge": [(2200, 2506), (3206, 1347), (2145, 376), (1073, 1425)],
             "black_edge": [(3566, 1368), (709, 1539), (725, 2563), (3571, 66)],
             "image_size": (2868, 4310)},
            "DSC_0083":
            {"square_ctr": (2148, 1456),
             "grating_edge": [(2071, 2149), (1441, 1593), (1856, 732), (2644, 2143), (2854, 1290),
                              (2586, 731), (1913, 736)],
             "white_edge": [(2200, 2506), (3206, 1347), (2145, 376), (1073, 1425)],
             "black_edge": [(3566, 1368), (709, 1539), (725, 2563), (3571, 66)],
             "image_size": (2868, 4310)},
            "DSC_0084":
            {"square_ctr": (2148, 1456),
             "grating_edge": [(2071, 2149), (1441, 1593), (1856, 732), (2644, 2143), (2854, 1290),
                              (2586, 731), (1913, 736)],
             "white_edge": [(2200, 2506), (3206, 1347), (2145, 376), (1073, 1425)],
             "black_edge": [(3566, 1368), (709, 1539), (725, 2563), (3571, 66)],
             "image_size": (2868, 4310)},
            "DSC_0085":
            {"square_ctr": (2148, 1456),
             "grating_edge": [(2071, 2149), (1441, 1593), (1856, 732), (2644, 2143), (2854, 1290),
                              (2586, 731), (1913, 736)],
             "white_edge": [(2200, 2506), (3206, 1347), (2145, 376), (1073, 1425)],
             "black_edge": [(3566, 1368), (709, 1539), (725, 2563), (3571, 66)],
             "image_size": (2868, 4310)},
            "DSC_0086":
            {"square_ctr": (2148, 1456),
             "grating_edge": [(2071, 2149), (1441, 1593), (1856, 732), (2644, 2143), (2854, 1290),
                              (2586, 731), (1913, 736)],
             "white_edge": [(2200, 2506), (3206, 1347), (2145, 376), (1073, 1425)],
             "black_edge": [(3566, 1368), (709, 1539), (725, 2563), (3571, 66)],
             "image_size": (2868, 4310)},
            "DSC_0087":
            {"square_ctr": (2148, 1456),
             "grating_edge": [(2071, 2149), (1441, 1593), (1856, 732), (2644, 2143), (2854, 1290),
                              (2586, 731), (1913, 736)],
             "white_edge": [(2200, 2506), (3206, 1347), (2145, 376), (1073, 1425)],
             "black_edge": [(3566, 1368), (709, 1539), (725, 2563), (3571, 66)],
             "image_size": (2868, 4310)}}

