#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python3 version of the official YCB download script.
Default: download a small demo subset into ./ycb. Adjust objects_to_download / files_to_download as needed.
"""

import os
import json
import urllib.request

output_directory = "./ycb"

# You can either set this to "all" or a list of the objects that you'd like to download.
# objects_to_download = "all"
objects_to_download = ["002_master_chef_can"]

# File types:
# 'berkeley_rgbd' : depth maps and images from Carmine cameras.
# 'berkeley_rgb_highres' : high-res images from Canon cameras.
# 'berkeley_processed' : segmented point clouds and textured meshes.
# 'google_16k' / 'google_64k' / 'google_512k' : google meshes.
# WARNING: rgbd/highres are large. Keep this small unless you intend to fetch many GB.
files_to_download = ["berkeley_processed"]  # default keep small; add "berkeley_rgbd" if needed

# Extract all files from the downloaded .tgz, and remove .tgz files.
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def fetch_objects(url):
    with urllib.request.urlopen(url) as response:
        data = response.read()
    objects = json.loads(data.decode("utf-8"))
    return objects["objects"]


def download_file(url, filename):
    with urllib.request.urlopen(url) as u, open(filename, "wb") as f:
        meta = u.info()
        file_size = int(meta.get("Content-Length", 0))
        print(f"Downloading: {filename} ({file_size/1e6:.2f} MB)")

        file_size_dl = 0
        block_sz = 65536
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if file_size:
                status = f"\r{file_size_dl/1e6:10.2f} MB  [{file_size_dl * 100. / file_size:3.2f}%]"
            else:
                status = f"\r{file_size_dl/1e6:10.2f} MB"
            print(status, end="")
        print()  # newline


def tgz_url(obj, file_type):
    if file_type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return f"{base_url}berkeley/{obj}/{obj}_{file_type}.tgz"
    elif file_type in ["berkeley_processed"]:
        return f"{base_url}berkeley/{obj}/{obj}_berkeley_meshes.tgz"
    else:
        return f"{base_url}google/{obj}_{file_type}.tgz"


def extract_tgz(filename, directory):
    os.system(f"tar -xzf {filename} -C {directory}")
    os.remove(filename)


def check_url(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req):
            return True
    except Exception:
        return False


if __name__ == "__main__":
    objects = objects_to_download  # or fetch_objects(objects_url) for all

    for obj in objects:
        if objects_to_download == "all" or obj in objects_to_download:
            for file_type in files_to_download:
                url = tgz_url(obj, file_type)
                if not check_url(url):
                    print(f"Skip {url} (not found)")
                    continue
                filename = f"{output_directory}/{obj}_{file_type}.tgz"
                download_file(url, filename)
                if extract:
                    extract_tgz(filename, output_directory)
