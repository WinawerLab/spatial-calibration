import os
from pyspacal import camera_data

configfile:
    os.path.expanduser("~/Documents/spatial-calibration/config.yml")

rule preprocess_image:
    input:
        os.path.join(config['DATA_DIR'], 'raw', '{filename}.NEF')
    output:
        os.path.join(config['DATA_DIR'], 'preprocessed', '{preproc_method}', '{filename}.{ext}')
    shell:
        "python -m pyspacal.utils {input} -p {wildcards.preproc_method}"

rule image_mtf:
    input:
        raw = os.path.join(config['DATA_DIR'], 'raw', '{filename}.NEF'),
        preproc1 = os.path.join(config['DATA_DIR'], 'preprocessed', 'no_demosaic', '{filename}.pgm'),
        preproc2 = os.path.join(config['DATA_DIR'], 'preprocessed', 'dcraw_vng_demosaic', '{filename}.tiff'),
        preproc3 = os.path.join(config['DATA_DIR'], 'preprocessed', 'dcraw_ahd_demosaic', '{filename}.tiff')
    output:
        os.path.join(config['DATA_DIR'], 'mtf-{filename}.csv')
    shell:
        "python -m pyspacal.mtf {input.raw} -f -p {output}"

rule join_csv:
    input:
        [os.path.join(config['DATA_DIR'], 'mtf-%s.csv' % f) for f, v in camera_data.IMG_INFO.items() if v[1] != 'blank']
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
