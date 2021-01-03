"""
"""
import urllib
import urllib.parse
import urllib.request

def send_http_request_get(url, params={}):
    #data = {}
    #for k in params:
    #    print(k)
    #    data[k] = urllib.parse.urlencode(params[k])
    data = urllib.parse.urlencode(params)
    #print(data)
    #print(url)
    response = urllib.request.urlopen(url+data)
    content = response.read()
    return content

def send_http_request_post(url, params={}):
    userAgent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    data = urllib.parse.urlencode(params)
    data = data.encode("utf-8")
    req = urllib.request.Request(url, data)
    response = urllib.request.urlopen(req)
    content = response.read()
    return content


