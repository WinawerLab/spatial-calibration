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
