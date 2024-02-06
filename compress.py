import pickle as pkl
import h5py
import netCDF4 as nc
import numpy as np
import xarray as xr

from pathlib import Path

def write_to_nc(data_pkl_path:Path, out_dir:Path):
    labels,coords,sflux = pkl.load(data_pkl_path.open("rb"))
    print(coords)
    print(labels)
    print(sflux.shape)
    da = xr.DataArray(name="sflux", data=sflux, dims=labels,
                      coords=coords[:-1]+[range(len(coords[-1]))],
                      attrs={"feature_labels":coords[-1]})
    ds = xr.Dataset(data_vars={"sflux":da}, coords=da.coords)
    ds.to_netcdf(
            path=out_dir.joinpath(data_pkl_path.stem+".nc"),
            mode="w",
            format="NETCDF4_CLASSIC",
            engine="netcdf4",
            )

if __name__=="__main__":
    data_pkl_path = Path("data/sflux_0.pkl")
    out_dir = Path("data")
    write_to_nc(data_pkl_path, out_dir)
