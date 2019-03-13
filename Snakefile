import os
from pyspacal import camera_data

configfile:
    os.path.expanduser("~/Documents/spatial-calibration/config.yml")


FIRST_PASS_IMGS = ['DSC_0239', 'DSC_0240', 'DSC_0241', 'DSC_0242', 'DSC_0243',
                   'DSC_0247', 'DSC_0248', 'DSC_0249', 'DSC_0250', 'DSC_0252',
                   'DSC_0254', 'DSC_0255', 'DSC_0256', 'DSC_0257', 'capt0000',
                   'capt0001', 'capt0002', 'capt0003', 'capt0004', 'capt0005',
                   'capt0006', 'capt0007', 'capt0008', 'capt0009', 'capt0010',
                   'capt0011', 'capt0012', 'capt0013', 'capt0014', 'capt0015',
                   'capt0019', 'capt0020', 'capt0021', 'capt0022', 'capt0023',
                   'capt0024', 'capt0025', 'capt0026', 'capt0027', 'capt0028',
                   'capt0029', 'capt0030', 'capt0032', 'capt0033', 'capt0034',
                   'capt0035', 'capt0042', 'capt0043', 'capt0044', 'capt0045']

def get_raw_img(wildcards):
    """some raw filenames end in .nef, some in .NEF
    """
    if wildcards.filename.startswith('capt'):
        ext = "nef"
    else:
        ext = "NEF"
    return os.path.join(config['DATA_DIR'], 'raw', '{filename}.%s' % ext)

rule preprocess_image:
    input:
        get_raw_img
    output:
        os.path.join(config['DATA_DIR'], 'preprocessed', '{preproc_method}', '{filename}.{ext}')
    shell:
        "python -m pyspacal.utils {input} -p {wildcards.preproc_method}"

rule image_mtf:
    input:
        raw = get_raw_img,
        preproc1 = os.path.join(config['DATA_DIR'], 'preprocessed', 'no_demosaic', '{filename}.pgm'),
        preproc2 = os.path.join(config['DATA_DIR'], 'preprocessed', 'dcraw_vng_demosaic', '{filename}.tiff'),
        preproc3 = os.path.join(config['DATA_DIR'], 'preprocessed', 'dcraw_ahd_demosaic', '{filename}.tiff')
    output:
        os.path.join(config['DATA_DIR'], 'mtf-{filename}.csv')
    shell:
        "python -m pyspacal.mtf {input.raw} -f -p {output}"

rule join_mtf_csv:
    input:
        [os.path.join(config['DATA_DIR'], 'mtf-%s.csv' % f) for f, v in camera_data.IMG_INFO.items() if v[2] not in ['blank', 'log_polar']]
    output:
        os.path.join(config['DATA_DIR'], 'mtf.csv')
    run:
        import pandas as pd
        import os

        df = []
        for f in input:
            df.append(pd.read_csv(f))
            os.remove(f)
        pd.concat(df).reset_index(drop=True).to_csv(output[0], index=False)


rule first_pass:
    input:
        raw = get_raw_img,
        preproc1 = os.path.join(config['DATA_DIR'], 'preprocessed', 'no_demosaic', '{filename}.pgm'),
        preproc2 = os.path.join(config['DATA_DIR'], 'preprocessed', 'dcraw_vng_demosaic', '{filename}.tiff'),
        preproc3 = os.path.join(config['DATA_DIR'], 'preprocessed', 'dcraw_ahd_demosaic', '{filename}.tiff')
    output:
        csv = os.path.join(config['DATA_DIR'], 'first_pass-{filename}-b{box_size_multiple}-s{step_size_multiple}.csv'),
        hdf5 = os.path.join(config['DATA_DIR'], 'first_pass-{filename}-b{box_size_multiple}-s{step_size_multiple}.hdf5')
    shell:
        "python -m pyspacal.first_pass {input.raw} -p {output.csv} -b {wildcards.box_size_multiple}"
        " -s {wildcards.step_size_multiple}"
        


rule join_first_pass_csv:
    input:
        [os.path.join(config['DATA_DIR'], 'first_pass-%s-b%s-s%s.csv' % (f, b, s)) for f in FIRST_PASS_IMGS for b in [1, 2] for s in [1, 2]]
    output:
        os.path.join(config['DATA_DIR'], 'first_pass.csv'),
        os.path.join(config['DATA_DIR'], 'first_pass.hdf5')
    run:
        import pandas as pd
        from pyspacal import DataFrameArray as dfa
        import os

        df = []
        for f in input:
            df.append(dfa.read_csv(f))
            os.remove(f)
            os.remove(os.path.splitext(f) + ".hdf5")
        pd.concat(df).reset_index(drop=True).to_csv(output[0], index=False)
