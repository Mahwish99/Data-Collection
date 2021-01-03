import base64
import StringIO
from PIL import Image

"""
"""
def byte2ndarray(bytestr):
    img = Image.open(StringIO.StringIO(unbase64str))
    if img.mode != "RGB":
        img = img.convert('RGB')
    img.flags.writeable = True
    return np.asarray(img)
