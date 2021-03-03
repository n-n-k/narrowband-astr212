import astroalign as aa
from astropy.io import fits

halpha_filepath = 'h-alpha/M82_h-alpha_120s_bin1_210126_071544_itzamna_seo_0_RAW.fits'
rband_filepath = 'r-band/M82_r-band_120s_bin1_210126_052734_itzamna_seo_0_RAW.fits'

rband_fits = fits.open(rband_filepath, do_not_scale_image_data=False)
halpha_fits = fits.open(halpha_filepath, do_not_scale_image_data=False)

rband = rband_fits[0].data
halpha = halpha_fits[0].data

registered_image, footprint = aa.register(halpha, rband)

new_fits = fits.HDUList([fits.PrimaryHDU(registered_image)])
new_fits.writeto('test.fits')

