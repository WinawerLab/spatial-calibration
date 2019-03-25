"""collect photos from tethered camera using gphoto

To use, plug your camera into the computer using a USB cable. Most operating systems will
automatically mount the camera; in order for gphoto2 to have access to it, you need to unmount
it. Do this by clicking eject in your file manager (but DO NOT physically unplug the camera). Then
run this script: `python gphoto.py`. As soon as you run it, the camera will capture an image,
transfer it to the computer in your current working directory as `capt0000.nef` (or whatever the
raw extension is for your camera), and delete it from the camera. Press Enter to gather the next
image. To finish, press `^C` (ctrl + c) to kill this program.

IMPORTANT: If you want to run this again, move the taken photos or rename them, because this
restarts its counter everytime the script is run.

"""
import subprocess

n = 0
while True:
    fname = 'capt%04d' % n
    subprocess.call(['gphoto2', '--capture-image-and-download', '--filename', fname + ".%C"])
    n += 1
    _ = input('Press Enter to continue...')
