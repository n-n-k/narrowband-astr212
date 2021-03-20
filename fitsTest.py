from astropy.io import fits
# from PIL import Image
import numpy as np
import cv2

def get_gradient(img):
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

halpha_filepath = 'M82_h-alpha_120s_bin1_210126_071544_itzamna_seo_0_RAW.fits'
rband_filepath = 'M82_r-band_120s_bin1_210126_052734_itzamna_seo_0_RAW.fits'

'''
hdul = fits.open(filepath, do_not_scale_image_data=False)
hdul.info()

data = hdul[0].data

cv2.imshow('test', data)
'''

rband_fits = fits.open(rband_filepath, do_not_scale_image_data=False)
halpha_fits = fits.open(halpha_filepath, do_not_scale_image_data=False)

rband = rband_fits[0].data
halpha = halpha_fits[0].data

rband_temp = np.array(rband).astype('float32')
halpha_temp = np.array(halpha).astype('float32')

#rband = cv2.cvtColor(rband, cv2.COLOR_BGR2GRAY)
#halpha = cv2.cvtColor(halpha, cv2.COLOR_BGR2GRAY)

'''
h, w = halpha.shape
print(h, w)

h, w = rband.shape
print(h, w)
'''

# cv2.imshow('rband', rband)
# cv2.imshow('halpha', halpha)

sz = rband.shape

warp_mode = cv2.MOTION_EUCLIDEAN
warp_matrix = np.eye(2, 3, dtype=np.float32)

num_iterations = 5000
termination_eps = 1e-10

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iterations, termination_eps)

cc, warp_matrix = cv2.findTransformECC(get_gradient(rband_temp), get_gradient(halpha_temp), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)

halpha_aligned = cv2.warpAffine(halpha, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

rband_resized = cv2.resize(rband, (512, 512))
halpha_resized = cv2.resize(halpha, (512, 512))
#halpha_aligned = cv2.resize(halpha_aligned, (512, 512))

print(halpha_aligned.shape)

#new_fits = fits.HDUList([fits.PrimaryHDU(halpha_aligned)])
#new_fits.writeto('test.fits')

raw_sum = rband + halpha
aligned_sum = rband + halpha_aligned

cv2.imshow('raw_sum', raw_sum)
cv2.imshow('aligned_sum', aligned_sum)

#cv2.imshow('rband', rband_resized)
#cv2.imshow('halpha', halpha_resized)
#cv2.imshow('aligned image halpha', halpha_aligned)

cv2.waitKey(0)

