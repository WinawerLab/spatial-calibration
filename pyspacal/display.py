#!/usr/bin/python
"""small script to test displays
"""

import argparse
from psychopy import visual, event
import numpy as np
from scipy import signal


def create_stimuli_set(screen_size):
    """create the stimuli set we want to use
    """
    num_freqs = int(np.floor(np.log2(np.min(screen_size)))) - 1
    img_size = 2 ** (num_freqs+1)
    print(img_size)
    x = np.linspace(0, 1, img_size, endpoint=False)
    x, y = np.meshgrid(x, x)
    im = np.zeros((num_freqs*3, img_size, img_size))
    mask = ((x-.5)**2 + (y-.5)**2) < (.25)**2
    white = ((x-.5)**2 + (y-.5)**2) < (.35)**2
    black = ((x-.5)**2 + (y-.5)**2) < (.45)**2
    black = black & ~white
    white = white & ~mask
    for i in range(num_freqs):
        f = 2**(i+1)
        # signal.square aliases at the high frequencies, and at those, np.cos looks almost exactly
        # like a square wave.
        if i >= num_freqs-3:
            func = np.cos
        else:
            func = signal.square
        im[i, :, :] = func(2*np.pi*f*x)
        im[i+num_freqs, :, :] = func(2*np.pi*f*y)
        im[i+num_freqs*2, :, :] = func(2*np.pi*f*(x*np.cos(np.pi/4) - y*np.sin(np.pi/4)))
    for i in range(len(im)):
        tmp = im[i, :, :] * mask
        tmp[white] = 1
        tmp[black] = -1
        im[i, :, :] = tmp
    return im


def test_display(screen_size, screen_num=1):
    """create a psychopy window and display some stimuli
    """
    if not hasattr(screen_size, "__iter__"):
        screen_size = [screen_size, screen_size]
    stimuli = create_stimuli_set(screen_size)
    win = visual.Window(screen_size, fullscr=True, screen=screen_num, colorSpace='rgb255',
                        color=127, units='pix')
    wait_text = visual.TextStim(win, ("Press q or esc to quit, any other key to advance"))
    win.mouseVisible = False
    win.gammaRamp = np.tile(np.linspace(0, 1, 256), (3, 1))
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
    parser.add_argument("--screen_num", "-s", type=int, default=1,
                        help="Which screen to display the images on")
    args = vars(parser.parse_args())
    test_display(**args)
