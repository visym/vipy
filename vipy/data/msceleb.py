import base64
import csv
import os
from vipy.util import remkdir, writecsv, readcsv


def extract(tsvfile, outdir):
    """https://github.com/cmusatyalab/openface/blob/master/data/ms-celeb-1m/extract.py"""
    with open(tsvfile, 'r') as tsvF:
        reader = csv.reader(tsvF, delimiter='\t')
        i = 0
        for row in reader:
            MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

            saveDir = os.path.join(outdir, MID)
            savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))

            remkdir(saveDir)
            with open(savePath, 'wb') as f:
                f.write(data)

            i += 1

            if i % 1000 == 0:
                print("Extracted {} images.".format(i))


def export(tsvfile, tsvnames, outdir, csvfile):
    csvlist = []
    d_mid_to_name = {x[0]:x[1] for x in readcsv(tsvnames, separator='\t')}
    with open(tsvfile, 'r') as tsvF:
        reader = csv.reader(tsvF, delimiter='\t')
        i = 0
        for row in reader:
            MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])

            saveDir = os.path.join(outdir, MID)
            savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))

            i += 1

            csvlist.append((savePath, d_mid_to_name[MID]))
            if i % 100 == 0:
                print("[msceleb.csv][%d]: Extracting CSV (%s,%s,%s)" % (i,savePath,MID,d_mid_to_name[MID]))

    print(writecsv(csvlist, csvfile))
