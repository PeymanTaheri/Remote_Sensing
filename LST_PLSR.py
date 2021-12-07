"""
Created on Tue Mar  5 09:36:58 2019

@author: Peyman Taheri
"""


import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import rasterio
import glob
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# getting the directory of landsat 8 dataset.
dirpath = r"Only\the\directoty\of\the\dataset"
search_criteria_raster = "L*.tif"
raster_q = os.path.join(dirpath, search_criteria_raster)
raster_fps = glob.glob(raster_q)
search_criteria_meta = "*MTL.txt"
meta_query = os.path.join(dirpath, search_criteria_meta)
meta = glob.glob(meta_query)
# sorting the rasters by band number.
raster_fps = sorted(raster_fps, key=lambda name: name[-8])
# Opening the metadata file and getting the requierd information for converting
# the DN numbers of optical bands to reflectance and converting the DN numbers of thermal band to radiance.
metadata = open(meta[0], 'r')

for line in metadata:

    if (line.find("REFLECTANCE_MULT_BAND") > 0):
        s = line.split("=")
        gain = float(s[-1])  # Getting the constant as float.

    elif (line.find("REFLECTANCE_ADD_BAND") > 0):
        s = line.split("=")
        bias = float(s[-1])

    elif (line.find("RADIANCE_MULT_BAND") > 0):
        s = line.split("=")
        band_num = int(s[0].split("_")[3])
        if band_num == 10:
            gain_10 = float(s[-1])

    elif (line.find("RADIANCE_ADD_BAND") > 0):
        s = line.split("=")
        band_num = int(s[0].split("_")[3])
        if band_num == 10:
            bias_10 = float(s[-1])

    elif (line.find("SUN_ELEVATION") > 0):
        s = line.split("=")
        teta = s[-1]

    elif (line.find("K1_CONSTANT_BAND") > 0):
        s = line.split("=")
        k1 = s[-1]

    elif (line.find("K2_CONSTANT_BAND") > 0):
        s = line.split("=")
        k2 = s[-1]


# Then stacking R, G, B, nir, SWIR1, SWIR2 and thermal band 10.
raster_stack = []
for fp in raster_fps:
    band_num = fp.split("_")[7]
    if band_num in ["B2.TIF", "B3.TIF", "B4.TIF", "B5.TIF", "B6.TIF", "B7.TIF", "B10.TIF"]:
        raster = rasterio.open(fp)
        raster_stack.append(raster)
# Copy the metadata file of the initial rasters to the new stack dataset
# metadata and updating count to the number of bands being stacked and saving the stacked file as stack.tif.
meta = raster_stack[0].meta
meta.update(count=len(raster_stack))
with rasterio.open('stack.tif', 'w', **meta) as dst:
    for id, layer in enumerate(raster_stack, start=1):
        dst.write_band(id, layer.read(1))

raster_stack = None
# Opening the saved stack dataset and croping it using shapefile and then saving it as "stack_clip.tif".
ds = gdal.Open("stack.tif")
ds_clip = gdal.Warp("stack_clip.tif", ds, cutlineDSName="name of the shapefile.shp",
                    cropToCutline=True, dstNodata=np.nan)
# Opening the clipped stack dataset and reading it as numpy
ds_c = gdal.Open("stack_clip.tif")
ds_crop = np.array(ds_c.ReadAsArray())

# Converting DN numbers of optical bands to reflectance by using the information acquired as mentioned above..
dim, row, col = ds_crop.shape
Ref = np.zeros((dim, row, col))


for i in range(6):
    Ref[i, :, :] = (ds_crop[i, :, :] * gain + bias) / \
        math.sin(math.radians(eval(teta)))

# Converting the DN numbers of thermal band to radiance.
L = ds_crop[dim-1, :, :]*gain_10 + bias_10
L_landa = (((np.nanmax(L) - np.nanmin(L))/(np.nanmax(ds_crop[dim-1, :, :]) -
                                           np.nanmin(ds_crop[dim-1, :, :]))) * (ds_crop[dim-1, :, :] - np.nanmin(ds_crop[dim-1, :, :]))) + np.nanmin(L)

# Calculating NDVI for emissivity estimation.
Ref[dim-1, :, :] = L_landa
nir = Ref[3, :, :]
red = Ref[2, :, :]
NDVI = (nir - red) / (nir + red)
pv = ((NDVI - 0.2)/(0.5 - 0.2)) * ((NDVI - 0.2)/(0.5 - 0.2))
Em = np.zeros((row, col))
# Estimating emissivity of each pixel by using NDVI threshold method.
for i in range(row):
    for j in range(col):
        if NDVI[i, j] > 0.5:
            Em[i, j] = 0.99

        elif 0.2 <= NDVI[i, j] <= 0.5:
            Em[i, j] = 0.00149 * pv[i, j] + 0.986

        elif NDVI[i, j] < 0.2:
            Em[i, j] = 0.97

# Calculating the brightness temperature of each thermal pixel
# First converting k1 and k2 constant to a matrice with the same size as clipped stack.
Bt = np.zeros((row, col))
x = np.zeros((row, col))
k_1 = np.zeros((row, col))
k_2 = np.zeros((row, col))

for i in range(row):
    for j in range(col):
        k_1[i, j] = k1

for i in range(row):
    for j in range(col):
        k_2[i, j] = k2

for i in range(row):
    for j in range(col):
        x[i, j] = k_1[i, j]/L_landa[i, j]

# Calculating Land Surface Temperature in celsius.
Bt = k_2 / np.log((k_1/L_landa) + 1)
landa = 10.8 * (10**-6)
h = 6.624 * (10**-34)
c = 2.998 * (10**8)
s = 1.38 * (10**-23)
c2 = h*c/s
y = 1 + ((landa * Bt)/c2) * np.log(Em)
LST = Bt / y - 273.15
LST[LST < 0] = np.nan
fig, ax = plt.subplots()
cax = ax.imshow(LST, cmap=cm.coolwarm)
ax.set_title('Land Surface Temperature')
cbar = fig.colorbar(cax, ticks=[np.nanmin(LST), np.nanmax(LST)])

# From here calculating other indecies such as UI and MNDWI and creating a panda data frame
# of NDVI, UI, MNDWI and LST for partial least square regression
# note that it is just a demonstration of how to preprocess raster datasets for PLSR.
gr = Ref[1, :, :]
sw1 = Ref[4, :, :]
sw2 = Ref[5, :, :]
UI = (sw2 - nir) / (sw2 + nir)
MNDWI = (gr - sw1) / (gr + sw1)
df = pd.DataFrame()
df['NDVI'] = NDVI.flatten()
df['UI'] = UI.flatten()
df['MNDWI'] = MNDWI.flatten()
df['LST'] = LST.flatten()
flate = LST.flatten()
df = df.dropna()
df_normal = df.copy()

for feature_name in df.columns:
    max_value = df[feature_name].max()
    min_value = df[feature_name].min()
    df_normal[feature_name] = (
        df[feature_name] - min_value) / (max_value - min_value)

X = df_normal.drop(columns=['LST'])
y = df_normal['LST']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Here the number of features are three as we only have NDVI, UI and MNDWI as features, so n_components is equal to three.
pls = PLSRegression(n_components=3)
pls.fit(X_train, y_train)
Y_pred = pls.predict(X_test)
print(pls.score(X_test, y_test))
plt.scatter(Y_pred, y_test, alpha=0.5)
plt.show()


# Refrences:
# [1] Sobrino, J.A.; Jiménez-Muñoz, J.C.; Paolini, L. Land surface temperature retrieval from landsat TM 5.
# Remote Sens. Environ. 2004, 90, 434–440.

# [2] Carlson, T.N.; Ripley, D.A. On the relation between NDVI, fractional vegetation cover, and leaf area index.
# Remote Sens. Environ. 1997, 62, 241–252.

# [3] Yu, X.; Guo, X.; Wu, Z. Land surface temperature retrieval from landsat 8 TIRS—Comparison between
# radiative transfer equation-based method, split window algorithm and single channel method. Remote Sens.
# 2014, 6, 9829–9852.

# [4] Artis, D.A.; Carnahan, W.H. Survey of emissivity variability in thermography of urban areas.
# Remote Sens. Environ. 1982, 12, 313–329.

# [5] Li, Z.-L.; Tang, B.-H.; Wu, H.; Ren, H.; Yan, G.; Wan, Z.; Trigo, I.F.; Sobrino, J.A. Satellite-derived land surface
# temperature: Current status and perspectives. Remote Sens. Environ. 2013, 131, 14–37.
