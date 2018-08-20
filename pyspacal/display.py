#!/usr/bin/python
"""small script to test displays
"""

import argparse
from psychopy import visual, event, monitors
import numpy as np
from scipy import signal

GAMMA_GRID = [[   0.591897  ,  157.111726  ,    2.40605342],
              [   0.591897  ,   49.805004  ,    2.35119282],
              [   0.591897  ,   92.823552  ,    2.44157325],
              [   0.591897  ,   15.467635  ,    2.66103173]]

def create_stimuli_set(img_size=512):
    """create the stimuli set we want to use
    """
    num_freqs = int(np.floor(np.log2(np.min(img_size)))) - 1
    print("Stimuli will contain %s pixels, gratings will be %s pixels wide" % (img_size,
                                                                               img_size//2))
    x = np.arange(-img_size//2, img_size//2)
    x, y = np.meshgrid(x, x)
    # our very last stimulus is just blank, so it will show mid-gray
    im = np.zeros((num_freqs*2+1, img_size, img_size))
    mask = np.max(np.dstack([np.abs(x), np.abs(y)]), 2) < img_size//4
    white = np.max(np.dstack([np.abs(x), np.abs(y)]), 2) < (img_size//4+img_size//8)
    black = np.max(np.dstack([np.abs(x), np.abs(y)]), 2) <= (img_size//2)
    black = black & ~white
    white = white & ~mask
    x = np.arange(0, img_size)
    x, _ = np.meshgrid(x, x)    
    for i in range(num_freqs):
        cycle_half_len = 2**(num_freqs-i-1)
        # signal.square can look weird due to precision issues
        # (https://stackoverflow.com/questions/38267021/incorrect-square-wave-sampling-using-scipy),
        # so we construct our own instead
        x = np.reshape(x, (img_size//cycle_half_len, img_size, cycle_half_len))
        x[:, ::2] = 1
        x[:, 1::2] = -1
        im[i, :, :] = np.reshape(x, (img_size, img_size)).copy()
        # these two variants give us the vertical and horizontal gratings
        x[::2] = 1
        x[1::2] = -1
        im[i+num_freqs, :, :] = np.reshape(x, (img_size, img_size)).copy()
    for i in range(len(im)-1):
        tmp = im[i, :, :] * mask
        tmp[white] = 1
        tmp[black] = -1
        im[i, :, :] = tmp
    return im


def test_display(screen_size, monitor_name='vpixx', img_size=512, screen_num=1):
    """create a psychopy window and display some stimuli
    """
    if not hasattr(screen_size, "__iter__") or len(screen_size) == 1:
        try:
            screen_size = [screen_size[0], screen_size[0]]
        except TypeError:
            screen_size = [screen_size, screen_size]
    assert img_size < screen_size[0], "Can't show larger image than the screen!"
    assert img_size < screen_size[1], "Can't show larger image than the screen!"
    stimuli = create_stimuli_set(img_size)
    win = visual.Window(screen_size, fullscr=True, screen=screen_num, colorSpace='rgb255',
                        color=127, units='pix')
    mon = monitors.Monitor(monitor_name)
    if monitor_name == 'vpixx':
        mon.setGamma(1)
    elif monitor_name == 'psychophysics_lcd':
        mon.setGammaGrid(np.array(GAMMA_GRID))
    else:
        raise Exception("I don't know your monitor's gamma and so the luminance values will be off."
                        " This might not be a huge deal -- if you don't care, set monitor_name to "
                        "vpixx to use a linear gamma")
    wait_text = visual.TextStim(win, ("Press q or esc to quit, any other key to advance"))
    win.mouseVisible = False
    wait_text.draw()
    win.flip()
    all_keys = event.waitKeys()
    if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
        win.close()
        return
    for i in range(len(stimuli)):
        thing_to_display = visual.ImageStim(win, stimuli[i], size=stimuli.shape[1:])
        thing_to_display.draw()
        win.flip()
        all_keys = event.waitKeys()
        if 'q' in [k[0] for k in all_keys] or 'escape' in [k[0] for k in all_keys]:
            break
    win.close()
    np.save('stim.npy', stimuli)
    # also get some info about the stimulus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test your display",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("screen_size", help="Screen size, in pixels. Can be one or two integers",
                        nargs='+', type=int)
    parser.add_argument('--monitor_name', '-m', default='vpixx',
                        help='Monitor name. Used to determine gamma values.')
    parser.add_argument("--img_size", "-i", default=512,
                        help="Image size, in pixels. Must be one integer", type=int)
    parser.add_argument("--screen_num", "-s", type=int, default=1,
                        help="Which screen to display the images on")
    args = vars(parser.parse_args())
    test_display(**args)
