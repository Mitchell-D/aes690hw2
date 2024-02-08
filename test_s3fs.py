
import numpy as np
import s3fs
import xarray as xr
from collections.abc import Mapping
from dask.distributed import Client
from datetime import datetime,timedelta
from itertools import chain
from pathlib import Path

def get_gmgsi(dataset, init_time:datetime, num_hours:int):
    """
    Query the GMGSI bucket for a continuous time range of hourly data

    :@param dataset: GMGSI dataset key; one of {"lw", "ssr", "sw", "vis", "wv"}
    :@param init_time: datetime object for the first hour of the desired range
    :@param num_hours: Integer number of hours from init_time for which data
        files will be returned

    :@return: (time,lat,lon) shaped xarray Dataset containing all files merged
        along the time axis, and with common lat/lon ordinates.
    """
    valid_dsets = {"lw", "ssr", "sw", "vis", "wv"}
    assert dataset in valid_dsets
    ## Construct a s3 path according to the gmgsi bucket structure
    bucket_path = f"s3://noaa-gmgsi-pds/GMGSI_{dataset.upper()}"
    ## Enumerate bucket paths for all valid hourly timesteps
    timesteps = [init_time+i*timedelta(hours=1) for i in range(num_hours)]
    ## Query the s3 bucket for all of the files
    bucket = s3fs.S3FileSystem(anon=True)
    bpath = lambda t:bucket_path+t.strftime("/%Y/%m/%d/%H/*")
    paths = sorted(list(chain(*[bucket.glob(bpath(t)) for t in timesteps])))
    files = list(map(bucket.open, paths))
    ## Open and aggregate the files along the time axis as an xarray Dataset
    return xr.open_mfdataset(files, combine='nested', concat_dim='time')

if __name__=="__main__":
    """ Load imagery for the full day of 20240115 """
    ds = get_gmgsi(dataset="lw", init_time=datetime(2024, 1, 15), num_hours=2)
    #np.save("tmp.npy", np.array(ds["data"]))

    """ Extract and average values over SEUS """
    lat0,latf = (25,35)
    lon0,lonf = (-75,-95)
    geo_mask = (ds["lat"][...]>=lat0) & (ds["lat"][...]<latf) & \
            (ds["lon"]>-lon0)[...] & (ds["lon"][...]>-lonf)
    geo_mask = np.broadcast_to(geo_mask, ds["data"].shape)
    ds = ds["data"].where(geo_mask)
    print(f"Average over SEUS: {np.nanmean(ds)}")
