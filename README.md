# ZMB-HCS

A collection of functions we use at [ZMB](https://www.zmb.uzh.ch/en.html) to handle HCS (high-content screening) data, and multiplexed data.

## Functionality
* Load and convert Molecular Devices ImageXpress acquisitions
* Calculate and apply shading corrections
* Stitch and fuse images
* Register images
* Collection of [fractal](https://fractal-analytics-platform.github.io/)-tasks (not yet tested in fractal!):
  * Converter for MD-ImageXpress data to ome-zarr
  * Calculate mean intensity projections
  * Calculate and perform illumination-correction
  * Calculate percentiles
  * Spot-segmentation
  * Measure features for hierarchical segmentations

## Installation

* It is recommended to install the package in a separate environment:
```
conda create -n zmb-hcs python=3.9 -y
conda activate zmb-hcs
```
* Install the package:
```
pip install git+https://github.com/ZMB-UZH/zmb-hcs
```
