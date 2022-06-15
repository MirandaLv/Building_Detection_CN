
from osgeo import gdal
import numpy as np
import os
import subprocess


def convert_to_8Bit(inputRaster, outputRaster,
                           outputPixType='Byte',
                           outputFormat='GTiff',
                           rescale_type='rescale',
                           percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit
    rescale_type = [clip, rescale]
        if clip, scaling is done strictly between 0 65535 
        if rescale, each band is rescaled to a min and max 
        set by percentiles
    '''
    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of', 
           outputFormat]
    
    # iterate through bands
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band = srcRaster.GetRasterBand(bandId)
        if rescale_type == 'rescale':
            bmin = band.GetMinimum()        
            bmax = band.GetMaximum()
            # if not exist minimum and maximum values
            if bmin is None or bmax is None:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            # else, rescale
            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(), 
                                 percentiles[0])
            bmax= np.percentile(band_arr_tmp.flatten(), 
                                percentiles[1])
        else:
            bmin, bmax = 0, 65535
            
        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))
    cmd.append(inputRaster)
    cmd.append(outputRaster)
    print("Conversin command:", cmd)
    subprocess.call(cmd)
    

    
path = "/rapids/notebooks/sciclone/geograd/EthanBrewer/SN2_buildings_train_AOI_4_Shanghai/AOI_4_Shanghai_Train/RGB-PanSharpen/"
files = os.listdir(path)

for file in files:
    resimPath = path+file
    dstPath   = "/rapids/notebooks/sciclone/geograd/EthanBrewer/SN2_buildings_train_AOI_4_Shanghai/AOI_4_Shanghai_Train/RGB-PanSharpen-NEW/"
    dstPath   = dstPath+file
    convert_to_8Bit(resimPath,dstPath)