import os
import platform
import subprocess
import time
from pathlib import Path

import requests
import torch


def gsutil_getsize(url=''):
    s = subprocess.check_output('gsutil du %s' % url, shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0


def attempt_download(weights):
    weights = str(weights).strip().replace("'", '')
    file = Path(weights).name.lower()

    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov3/releases/'
    response = requests.get('https://api.github.com/repos/ultralytics/yolov3/releases/latest').json()
    assets = [x['name'] for x in response['assets']]
    redundant = False

    if file in assets and not os.path.isfile(weights):
        try:
            tag = response['tag_name']
            url = f'https://github.com/ultralytics/yolov3/releases/download/{tag}/{file}'
            print('Downloading %s to %s...' % (url, weights))
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6
        except Exception as e:
            print('Download error: %s' % e)
            assert redundant, 'No secondary mirror'
            url = 'https://storage.googleapis.com/ultralytics/yolov3/ckpt/' + file
            print('Downloading %s to %s...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):
                os.remove(weights) if os.path.exists(weights) else None
                print('ERROR: Download failure: %s' % msg)
            print('')
            return


def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', name='tmp.zip'):
    t = time.time()
    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    os.remove(name) if os.path.exists(name) else None
    os.remove('cookie') if os.path.exists('cookie') else None

    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system('curl -c ./cookie -s -L "drive.google.com/uc?export=download&id=%s" > %s ' % (id, out))
    if os.path.exists('cookie'):
        s = 'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm=%s&id=%s" -o %s' % (get_token(), id, name)
    else:
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    r = os.system(s)
    os.remove('cookie') if os.path.exists('cookie') else None

    if r != 0:
        os.remove(name) if os.path.exists(name) else None
        print('Download error ')
        return r

    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)
        os.remove(name)

    print('Done (%.1fs)' % (time.time() - t))
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""