import os
from pyspacal import camera_data

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data')

FIRST_PASS_IMGS = ['DSC_0239', 'DSC_0240', 'DSC_0241', 'DSC_0242', 'DSC_0243', 'DSC_0247',
                   'DSC_0248', 'DSC_0249', 'DSC_0250', 'DSC_0252', 'DSC_0254', 'DSC_0255',
                   'DSC_0256', 'DSC_0257', 'capt0003', 'capt0004', 'capt0005', 'capt0006',
                   'capt0007', 'capt0011', 'capt0012', 'capt0013', 'capt0014', 'capt0015',
                   'capt0022', 'capt0023', 'capt0024', 'capt0025', 'capt0026', 'capt0030',
                   'capt0032', 'capt0033', 'capt0034', 'capt0035', 'capt0042', 'capt0043',
                   'capt0044', 'capt0045']

rule download_data:
    output:
        [os.path.join(DATA_DIR, 'raw', f'{fname}.{"nef" if fname.startswith("capt") else "NEF"}') for fname in camera_data.IMG_INFO.keys()]
    params:
        tar_path = 'prisma_raw_images.tar.gz',
        dir_name = lambda wildcards, output: os.path.dirname(output[0]),
    shell:
        "curl -O -J -L https://osf.io/83fna/download; "
        "tar xf {params.tar_path} -C {params.dir_name}; "
        "rm {params.tar_path}"


def get_raw_img(wildcards):
    """some raw filenames end in .nef, some in .NEF
    """
    if wildcards.filename.startswith('capt'):
        ext = "nef"
    else:
        ext = "NEF"
    return os.path.join(DATA_DIR, 'raw', '{filename}.%s' % ext)

rule preprocess_image:
    input:
        get_raw_img,
    output:
        os.path.join(DATA_DIR, 'preprocessed', '{preproc_method}', '{filename}.{ext}')
    shell:
        "python -m pyspacal.utils {input} -p {wildcards.preproc_method}"

# these are 3 separate rules because they will confuse each other: the intermediate outputs created
# for the vng and ahd demosaic have the same filename (and I can't figure out how to change that),
# so they cannot be run simultaneously (if you're only running 1 job at a time, then this doesn't
# matter).
rule preprocess_all_vng:
    input:
        [os.path.join(DATA_DIR, 'preprocessed', 'dcraw_vng_demosaic', '%s.tiff' % f) for f, v in camera_data.IMG_INFO.items() if v[2] != 'log_polar'],

rule preprocess_all_ahd:
    input:
        [os.path.join(DATA_DIR, 'preprocessed', 'dcraw_ahd_demosaic', '%s.tiff' % f) for f, v in camera_data.IMG_INFO.items() if v[2] != 'log_polar'],

rule preprocess_all_no_demosaic:
    input:
        [os.path.join(DATA_DIR, 'preprocessed', 'no_demosaic', '%s.pgm' % f) for f, v in camera_data.IMG_INFO.items() if v[2] != 'log_polar'],

rule image_mtf:
    input:
        raw = get_raw_img,
        preproc1 = os.path.join(DATA_DIR, 'preprocessed', 'no_demosaic', '{filename}.pgm'),
        preproc2 = os.path.join(DATA_DIR, 'preprocessed', 'dcraw_vng_demosaic', '{filename}.tiff'),
        preproc3 = os.path.join(DATA_DIR, 'preprocessed', 'dcraw_ahd_demosaic', '{filename}.tiff')
    output:
        os.path.join(DATA_DIR, 'mtf-{filename}.csv')
    shell:
        "python -m pyspacal.mtf {input.raw} -f -p {output}"

rule join_mtf_csv:
    input:
        [os.path.join(DATA_DIR, 'mtf-%s.csv' % f) for f, v in camera_data.IMG_INFO.items() if v[2] not in ['blank', 'log_polar']]
    output:
        os.path.join(DATA_DIR, 'mtf.csv')
    run:
        import pandas as pd
        import os

        df = []
        for f in input:
            df.append(pd.read_csv(f))
            os.remove(f)
        pd.concat(df).reset_index(drop=True).to_csv(output[0], index=False)


rule mtf_spline:
    input:
        os.path.join(DATA_DIR, 'mtf.csv')
    output:
        os.path.join(DATA_DIR, 'mtf-spline.svg'),
        os.path.join(DATA_DIR, 'mtf-spline.pkl'),
    run:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import interpolate
        import pickle
        df = pd.read_csv(input[0])
        constraints = {'image_context': 'projector_4', 'grating_type':
                       'grating', 'image_source': 'demosaiced_image_greyscale',
                       'preprocess_type': 'dcraw_vng_demosaic',
                       'contrast_type': 'michelson'}
        tmp = df[np.all([df[k]==v for k,v in constraints.items()], 0)]
        # this averages across the two grating directions, vertical and horizontal
        mtf_vals = tmp.groupby('display_freq')['corrected_contrast'].mean()

        sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
        s = interpolate.UnivariateSpline(mtf_vals.index, mtf_vals.values,k=1)
        xnew = np.linspace(2**-9, 2**-1)
        with sns.plotting_context('poster'):
            plt.figure(figsize=(7,5))
            plt.semilogx(mtf_vals, 'o', basex=2)
            plt.semilogx(xnew, s(xnew), basex=2);
            plt.xlabel('Display frequency (cpp)')
            plt.ylabel('Michelson Contrast')
            plt.xticks([2**-9, 2**-7, 2**-5, 2**-3, 2**-1])
            plt.savefig(output[0], bbox_inches='tight')

        with open(output[1], 'wb') as f:
            pickle.dump(s, f)


rule first_pass:
    input:
        raw = get_raw_img,
        preproc1 = os.path.join(DATA_DIR, 'preprocessed', 'no_demosaic', '{filename}.pgm'),
        preproc2 = os.path.join(DATA_DIR, 'preprocessed', 'dcraw_vng_demosaic', '{filename}.tiff'),
        preproc3 = os.path.join(DATA_DIR, 'preprocessed', 'dcraw_ahd_demosaic', '{filename}.tiff')
    output:
        csv = os.path.join(DATA_DIR, 'first_pass-{filename}-b{box_size_multiple}-s{step_size_multiple}.csv'),
        hdf5 = os.path.join(DATA_DIR, 'first_pass-{filename}-b{box_size_multiple}-s{step_size_multiple}.hdf5')
    shell:
        "python -m pyspacal.first_pass {input.raw} -p {output.csv} -b {wildcards.box_size_multiple}"
        " -s {wildcards.step_size_multiple}"
        


rule join_first_pass_csv:
    input:
        [os.path.join(DATA_DIR, 'first_pass-%s-b%s-s%s.csv' % (f, b, s)) for f in FIRST_PASS_IMGS for b in [1, 2] for s in [1, 2]]
    output:
        os.path.join(DATA_DIR, 'first_pass.csv'),
        os.path.join(DATA_DIR, 'first_pass.hdf5')
    run:
        import pandas as pd
        from pyspacal import DataFrameArray as dfa
        import os

        df = []
        for f in input:
            df.append(dfa.read_csv(f))
            os.remove(f)
            os.remove(os.path.splitext(f)[0] + ".hdf5")
        pd.concat(df).reset_index(drop=True).to_csv(output[0], index=False)
