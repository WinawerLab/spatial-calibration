import os
import pandas as pd

configfile:
    "/home/billbrod/Documents/spatial-calibration/config.yml"

rule preprocess_image:
    input:
        os.path.join(config['DATA_DIR'], 'raw', '{filename}.NEF')
    output:
        os.path.join(config['DATA_DIR'], 'preprocessed', '{preproc_method}', '{filename}.{ext}')
    shell:
        "python pyspacal/utils.py {input} -p {preproc_method}"

rule image_mtf:
    input:
        raw = os.path.join(config['DATA_DIR'], 'raw', '{filename}.NEF'),
        preproc1 = os.path.join(config['DATA_DIR'], 'preprocessed', 'no_demosaic', '{filename}.pgm'),
        preproc2 = os.path.join(config['DATA_DIR'], 'preprocessed', 'dcraw_vng_demosaic', '{filename}.tiff'),
        preproc3 = os.path.join(config['DATA_DIR'], 'preprocessed', 'dcraw_ahd_demosaic', '{filename}.tiff')
    output:
        os.path.join(config['DATA_DIR'], 'mtf-{filename}.csv')
    shell:
        "python pyspacal/mtf.py {input.raw} -f -s {output}"
