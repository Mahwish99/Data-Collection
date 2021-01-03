import os
import base64


def file2base64str(filepath):
    if os.path.exists(filepath) == False:
        print("[Error]: file no exist!")
        return None
    pfile = open(filepath, "rb")
    text64str = base64.b64encode(pfile.read())
    pfile.close()
    return text64str

def base64str2file(encodingstr, filepath):
    try:
        decodestr = base64.b64decode(encodingstr)
        pfile = open(filepath, "wb")
        pfile.write(decodestr)
        pfile.close()
    except Exception as e:
        print(e)
        return 1
    return 0

