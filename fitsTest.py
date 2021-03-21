"""

Written by Naren Kasinath

This code calculates the shift between a broadband and narrowband image
and creates a new narrowband image with the shift applied to its WCS

"""

# Imports
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from copy import deepcopy
import cv2
import math
import numpy as np


def make_triangles (points):
    """
    Recursively create a list of triangles from a given set of points

    Parameters:
        points (list) : the list of Cartesian points

    Returns:
        triangles (list) : the list of triangle vertices
    """
    # Initialize the triangles array
    triangles = []
    # Get the first two vertices and the remaining tertiary points
    first, second, rest = points[0], points[1], points[2:]

    # Iterate through the remaining tertiary points to make triangles
    for point in rest:
        triangles.append((first, second, point))
    
    # Construct the next set of points to iterate through
    next_points = [second] + rest
    
    # If there are more than 2 points, recrusively make more triangles
    # otherwise, cease recursion and return the current list
    if len(next_points) > 2:
        return triangles + make_triangles(next_points)
    else:
        return triangles


def image_proc(path):
    """
    Construct and process an image from a given filepath

    Parameters:
        path (str) : the filepath to the image

    Returns:
        image (numpy.ndarray) : the raw image
        image_clipped (numpy.ndarray) : the sigma_clipped image
        blur (np.ndarray) : the median-blurred image
        contours (list) : the list of contours
        stars (list) : the list of identified star coordinates
        radii (list) : the list of identified star radii
        triangles (list) : the list of vertices for the star triangles
    """
    # Open and construct the file from the filepath
    image = fits.open(path, do_not_scale_image_data=False)[0].data
    # Make a deepcopy of the image to avoid damaging the original file
    image_clipped = deepcopy(image)
    
    # Iterate through each row of the image for sigma clipping
    for row in image_clipped:
        # Get the mask array for the row
        row_mask = sigma_clip(row, masked=True).recordmask
        # Mask each star and object with a value of 1 and background with 0
        star_indices = row_mask == True
        bg_indices = row_mask == False
        row[star_indices] = 1
        row[bg_indices] = 0

    # Rescale the image to uint8 for use with later functions
    image_clipped = np.array(image_clipped * 255, np.uint8)
    
    # Apply a Median Blur with a 15x15 kernel
    blur = cv2.medianBlur(image_clipped, 15)
    
    # Find edge contours in the blurred image
    contours, hierarchy = cv2.findContours(blur,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Prepare arrays for identified stars and their radii
    stars = []
    radii = []

    # Iterate through the list of contours
    for contour in contours:
        # Calculate the smallest possible circle that encloses the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Check if the contour and the circle differ in radii by 10px at most
        # This filters nearly all objects and keeps only stars
        if abs((np.pi * radius**2) - cv2.contourArea(contour)) < 100:
            stars.append(center)
            radii.append(radius)

    # Create a list of triangles with stars as vertices
    triangles = []

    if len(stars) > 2:
        triangles = make_triangles(stars)
        
    return image, image_clipped, blur, contours, stars, radii, triangles


def calc_shift(reference_triangles, narrowband_triangles):
    """
    Calculates the shift between a reference image and a narrowband image

    Parameters:
        reference_triangles (list) : list of reference triangle vertices
        narrowband_triangles (list) : list of narrowband triangle vertices

    Returns:
        shift (tuple) : x, y shift in px
        matched_triangle_indices (list) : list of tuples of matched triangle
                                          indices
    """
    shift = None
    matched_triangle_indices = []
    dxs = []
    dys = []
    
    # Iterate through each reference triangle
    for i, reference_triangle in enumerate(reference_triangles):
        # Sort the vertices by y-coordinate, then x-coordinate
        reference_triangle = sorted(reference_triangle,
                                    key= (lambda k: [k[1], k[0]]))
        r_a = reference_triangle[0]
        r_b = reference_triangle[1]
        r_c = reference_triangle[2]

        # Compute side lengths and triangle area
        reference_sides = [math.dist(r_a, r_b),
                           math.dist(r_b, r_c),
                           math.dist(r_a, r_c)]
        reference_area = abs(((r_a[0] * (r_b[1] - r_c[1])) +
                              (r_b[0] * (r_c[1] - r_a[1])) +
                              (r_c[0] * (r_a[1] - r_b[1])))
                             / 2)
        
        # Iterate through each narrowband triangle to compare to the reference
        for j, narrowband_triangle in enumerate(narrowband_triangles):
            # Sort the vertices by y-coordinate, then x-coordinate
            narrowband_triangle = sorted(narrowband_triangle,
                                         key= (lambda k: [k[1], k[0]]))
            n_a = narrowband_triangle[0]
            n_b = narrowband_triangle[1]
            n_c = narrowband_triangle[2]

            # Compute side lengths and triangle area
            narrowband_sides = [math.dist(n_a, n_b),
                                math.dist(n_b, n_c),
                                math.dist(n_a, n_c)]
            narrowband_area = abs(((n_a[0] * (n_b[1] - n_c[1])) +
                                   (n_b[0] * (n_c[1] - n_a[1])) +
                                   (n_c[0] * (n_a[1] - n_b[1])))
                                  / 2)
            
            # If the difference between side lengths, area, and
            # point positions is within the threshold, it is a match
            # Note: This threshold is completely arbitrary so far
            if abs(narrowband_sides[0] - reference_sides[0]) < 50 and \
               abs(narrowband_sides[1] - reference_sides[1]) < 50 and \
               abs(narrowband_sides[2] - reference_sides[2]) < 50 and \
               abs(narrowband_area - reference_area) < 2000 and \
               math.dist(n_a, r_a) < 50 and \
               math.dist(n_b, r_b) < 50 and \
               math.dist(n_c, r_c) < 50:
                   # Keep the indices of the triangles along with x & y shifts
                   matched_triangle_indices.append((i, j))
                   dxs.append(np.average([n_a[0] - r_a[0],
                                          n_b[0] - r_b[0],
                                          n_c[0] - r_c[0]]))
                   dys.append(np.average([n_a[1] - r_a[1],
                                          n_b[1] - r_b[1],
                                          n_c[1] - r_c[1]]))
    
    # Compute the average x & y shift
    if len(dxs) > 0:
        shift = (np.average(dxs), np.average(dys))

    return shift, matched_triangle_indices


def shift_narrowband(reference_path, narrowband_path, shift, new_path):
    """
    Find the WCS coordinates of the narrowband image by shifting those of the
    reference image

    Parameters:
        reference_path (str) : path to the reference image
        narrowband_path (str) : path to the narrowband image
        shift (tuple) : x & y shift
        new_path (str) : path to the new narrowband image to be created

    Returns:
        updated_narrowband (numpy.ndarray) : the narrowband image with shifted
                                             WCS information
    """
    # Initialize the reference WCS system from the reference FITS header
    reference_header = fits.open(reference_path)[0].header
    reference_wcs = WCS(reference_header)

    # Load and create copies of the narrowband image and FITS header
    narrowband_image = deepcopy(fits.open(narrowband_path)[0].data)
    narrowband_header = deepcopy(fits.open(narrowband_path)[0].header)
    
    # Compute the angular shift of the origin in the reference WCS system
    x, y = reference_wcs.pixel_to_world_values(1024 + shift[0],
                                               1024 + shift[1])
    # Set the RA and DEC values of the origin in the narrowband image to
    # the shifted values
    narrowband_header.set('CRVAL1', value=float(x))
    narrowband_header.set('CRVAL2', value=float(y))

    # Construct and save the adjusted FITS file
    updated_narrowband = fits.PrimaryHDU(data = narrowband_image,
                                         header = narrowband_header)
    updated_narrowband.writeto(new_path, overwrite=True)

    return updated_narrowband


def make_displayable_image(image, stars=None, radii=None, triangles=None):
    """
    Make a displayable image

    Parameters:
        image (numpy.ndarray) : the input grayscale image
        stars (list) : the list of star coordinates
        radii (list) : the list of star radii
        triangles (list) : the list of star triangle vertices

    Returns:
        displayable_image (numpy.ndarray) : the displayable image
    """
    # First convert the grayscale image to BGR color
    displayable_image = cv2.cvtColor(deepcopy(image), cv2.COLOR_GRAY2BGR)

    # If the stars and radii are supplied, draw corresponding outline circles
    if stars is not None and radii is not None:
        for i in range(0, len(stars)):
            displayable_image = cv2.circle(displayable_image,
                                           stars[i],
                                           radii[i],
                                           (255, 0, 0),
                                           2)

    # If the triangles are supplied, draw corresponding outline triangles
    if triangles is not None:
        for triangle in triangles:
            displayable_image = cv2.polylines(displayable_image,
                                              [np.array(triangle, np.int32)],
                                              True,
                                              (0, 0, 255),
                                              2)
    
    return displayable_image


def highlight_triangle(image, triangles, index):
    """
    Highlight a specific triangle for an image

    Parameters:
        image (numpy.ndarray) : the input image
        triangles (list) : the list of triangle vertices
        index : the index of the desired triangle
    """
    # Copy the input image
    displayable_image = deepcopy(image)
    # Retrieve the desired triangle and draw it with a unique red color
    triangle = triangles[index]
    displayable_image = cv2.polylines(displayable_image,
                                      [np.array(triangle, np.int32)],
                                      True,
                                      (0, 255, 0),
                                      2)
    return displayable_image


# Some example code for image processing

# rband and halpha filepaths
rband_filepath = 'M82_r-band_120s_bin1_210126_052734_itzamna_seo_0_FCAL.fits'
halpha_filepath = 'M82_h-alpha_120s_bin1_210126_071544_itzamna_seo_0_HPX.fits'

# Process the rband and halpha images
rband, rband_clipped, rband_blur, rband_contours, rband_stars, rband_radii, rband_triangles = image_proc(rband_filepath)

halpha, halpha_clipped, halpha_blur, halpha_contours, halpha_stars, halpha_radii, halpha_triangles = image_proc(halpha_filepath)

# Compute the shift between the halpha and rband image
shift, triangle_inds = calc_shift(rband_triangles, halpha_triangles)

# Create and save the halpha image with updated WCS coordinates
updated_halpha = shift_narrowband(rband_filepath, 
                                  halpha_filepath, 
                                  shift, 
                                  'M82_h-alpha_120s_bin1_120_21026_071544_itzamna_seo_0_WCS.fits')

# Create display images for rband and halpha
# Only draw matching triangles
rband_image = make_displayable_image(rband_blur,
                                     rband_stars,
                                     rband_radii)
rband_image = highlight_triangle(rband_image,
                                 rband_triangles,
                                 triangle_inds[0][0])

halpha_image = make_displayable_image(halpha_blur,
                                      halpha_stars,
                                      halpha_radii)
halpha_image = highlight_triangle(halpha_image,
                                  halpha_triangles,
                                  triangle_inds[0][1])

cv2.imshow('rband', rband_image)
cv2.imshow('halpha', halpha_image)

cv2.waitKey(0)

