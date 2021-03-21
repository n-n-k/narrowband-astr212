import astroalign as aa
from astropy.io import fits
import numpy as np
import cv2
from PIL import Image
import time
import requests
import json
import random
from urllib.parse import urlencode, quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError
OUTPUTFILE = "aaedBroadband.png"

halpha_filepath = '../M82_h-alpha_120s_bin1_210126_071544_itzamna_seo_0_RAW.fits'
rband_filepath = '../M82_r-band_120s_bin1_210126_052734_itzamna_seo_0_RAW.fits'

rband_fits = fits.open(rband_filepath, do_not_scale_image_data=False)
halpha_fits = fits.open(halpha_filepath, do_not_scale_image_data=False)

rband16 = rband_fits[0].data
halpha16 = halpha_fits[0].data

rband8 = cv2.normalize(rband16, None, 0, 255,
                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)

halpha8 = cv2.normalize(halpha16, None, 0, 255,
                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)

rband16Rgb1 = cv2.cvtColor(rband16, cv2.COLOR_BayerGB2RGB)
halpha16Rgb1 = cv2.cvtColor(halpha16, cv2.COLOR_BayerGB2RGB)

cv2.imwrite('source.png', rband16Rgb1)
cv2.imwrite('target.png', halpha16Rgb1)

# for reference: aa.register(source, target)
# we are transforming the broadband image to match the narrowband
registered, footprint = aa.register(rband16Rgb1, halpha16Rgb1)
time.sleep(3)
cv2.imwrite(OUTPUTFILE, np.uint8(registered))

time.sleep(3)
print("Preparing to complete WCS solve...")

# connect to api, get token
apiKey = "oppqxlkuubsdwsok"  # Key from Simon Mahns
R = requests.post('http://nova.astrometry.net/api/login',
                  data={'request-json': json.dumps({"apikey": apiKey})})
token = json.loads(R.text)["session"]
print(token)

#########
payload2 = {"publicly_visible": "y", "allow_modifications": "d",
            "session": token, "allow_commercial_use": "d"}
#########


boundary_key = ''.join(
    [random.choice('0123456789') for i in range(19)])
boundary = '===============' + str(boundary_key) + '=='
print(type(boundary))
headers = {'Content-Type':
           'multipart/form-data; boundary=' + boundary}
# data_pre = ('--' + str(boundary) + '\n' + 'Content-Type: text/plain\r\n' + 'MIME-Version: 1.0\r\n' + 'Content-disposition: form-data; name="request-json"\r\n' + '\r\n' + json + '\n' + '--' +
#             boundary + '\n' + 'Content-Type: application/octet-stream\r\n' + 'MIME-Version: 1.0\r\n' + 'Content-disposition: form-data; name="file"; filename="' + OUTPUTFILE + '"' + '\r\n' + '\r\n')
data_pre2 = "--" + boundary + "\n" + \
    "Content-Type: text/plain\r\n" + "MIME-Version: 1.0\r\n" + \
    'Content-disposition: form-data; name="request-json"\r\n' + \
    "\r\n" + str(json.dumps(payload2)) + '\n' + "--" + boundary + '\n' + 'Content-Type: application/octet-stream\r\n' + \
    'MIME-Version: 1.0\r\n' + 'Content-disposition: form-data; name="file"; filename="' + \
    OUTPUTFILE + '"' + '\r\n' + '\r\n'
print(data_pre2)

data_post = ('\n' + '--' + boundary + '--\n')
data = data_pre2.encode() + OUTPUTFILE.encode() + data_post.encode()

resp = Request(url="http://nova.astrometry.net/api/upload",
               headers=headers, data=data)
# post image
# resp = requests.post('http://nova.astrometry.net/api/url_upload',
#
#                data=data)
f = urlopen(resp)
txt = f.read()
subid = json.loads(txt)["subid"]

print(subid)

# wait
print("waiting for results...")
time.sleep(100)


# check if job is done
resp = requests.get('http://nova.astrometry.net/api/submissions/' +
                    str(subid), data={'request-json': json.dumps({"session": token})})
if(len(json.loads(resp.text)["job_calibrations"]) == 0):
    print("still waiting for results...")
    time.sleep(200)

resp = requests.get('http://nova.astrometry.net/api/submissions/' +
                    str(subid), data={'request-json': json.dumps({"session": token})})
# get jobid
jobid = json.loads(resp.text)["jobs"][0]
print(jobid)


# get calibration
resp = requests.get('http://nova.astrometry.net/api/jobs/' + str(jobid) +
                    '/info/', data={'request-json': json.dumps({"session": token})})
print(resp.text)
