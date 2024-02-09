import json
import pickle as pkl
import h5py
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
from pathlib import Path

def write_to_nc(name:str, labels:list, coords:list, data:np.ndarray,
                out_path:Path, use_xarray=False, compression=None,
                dtype="f8", complevel=4, sig_digits=None, round_digits=None):
    """
    Compression options:
    zlib,szip,zstd,bzip2,blosc_lz,blosc_lz4,blosc_lz4hc,blosc_zlib,blosc_zstd

    --(keyword parameters only have an effect if use_xarray=False)--

    :@param use_xarray: If True, generates the netCDF using the xarray format,
        and with the xarray backend.
    :@param compression: Algorithm; None or one of the string options above.
    :@param dtype: netCDF-syntax data type (ie "f4", "u8", "i2")
    :@param complevel: Compression in [1,9]; Ignored if using szip
    :@param sig_digits: No quantization performed if set to None.
        If set to a positive integer, BitGroom quantization is used to compress
        the data. The value determines the number of significant digits
        retained in the float mantissa, which has a literal accuracy depending
        on the value of the exponent bit.
    """
    ## By convention, the feature axis with string labels is last.
    ## Replace the string labels with an index range, and store the labels
    ## as an attribute.
    flabels = coords[-1]
    coords = list(map(list,coords[:-1]+[range(len(coords[-1]))]))
    if use_xarray:
        da = xr.DataArray(name=name, data=sflux, dims=labels, coords=coords)
        da.attrs.update({"flabels":flabels})
        ds = xr.Dataset(data_vars={name:da}, coords=da.coords)
        ds.to_netcdf(path=out_path,mode="w",format="NETCDF4",engine="netcdf4")
        return

    ## Round the data if requested
    if not round_digits is None:
        data = np.round(data, decimals=round_digits)

    ## Initialize a netCDF dataset with variables for the data and coords
    dataset = nc.Dataset(out_path, "w", format="NETCDF4")
    group = dataset.createGroup("data")
    group.flabels = tuple(flabels)
    dims = [group.createDimension(labels[i],len(coords[i]))
            for i in range(len(labels))]
    data_var = group.createVariable(
            name,
            datatype=dtype,
            dimensions=dims,
            compression=compression,
            complevel=complevel,
            quantize_mode="BitGroom",
            significant_digits=sig_digits,
            )
    coord_vars = [
            group.createVariable(
                labels[i],  datatype=dtype, dimensions=[dims[i]])
            for i in range(len(labels))
            ]

    ## Load the coordinates and data into the netCDF
    for i in range(len(coord_vars)):
        coord_vars[i][:] = np.array(coords[i])
    data_var[:] = data

def load_from_nc(name:str, nc_path:Path):
    """
    Re-loads the data from a file writen by write_to_nc, and returns the
    labels, coordinates, and data in their original form.

    :@param name: Main data array name in netCDF dataset
    :@param nc_path: Path to netCDF file created by write_to_nc
    """
    dataset = nc.Dataset(nc_path)
    group = dataset["data"]
    labels = list(group.dimensions.keys())
    coords = [np.array(group[l]) for l in labels]
    coords[-1] = group.flabels
    data = np.array(group[name])
    return labels,coords,data

def plot_sizes(results:dict, byte_scale=1e9, plot_spec={}):
    fig,ax = plt.subplots()
    labels,sizes = zip(*results.items())
    sizes = np.array(sizes)/byte_scale
    ax.grid(zorder=0,axis="y")
    ax.bar(labels,sizes, zorder=3)
    ax.set_title(plot_spec.get("title"))
    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    plt.show()
    print(list(zip(labels,list(np.amax(sizes)/sizes))))

if __name__=="__main__":
    data_pkl_path = Path("data/sflux_1.pkl")
    out_dir = Path("data")
    dataset_name = "sflux"
    ## JSON where compressed file size results are stored
    sizes_path = out_dir.joinpath(f"{dataset_name}_sizes.json")

    """
    Load axis labels, coordinate arrays, and the lookup table from the pkl
    generated by get_lut.py
    """
    labels,coords,data = pkl.load(data_pkl_path.open("rb"))
    flabels = coords[-1]
    print(flabels)
    print(labels)
    print(data.shape, data.size)
    print(data.dtype)
    print(tuple(zip(labels,[c[0] for c in coords])))

    """ Define netCDF configurations """
    run_args = {
            "f8":{"dtype":"f8"},
            "f4":{"dtype":"f4"},
            "zlib":{"dtype":"f8", "compression":"zlib"},
            "q1":{"sig_digits":1, "compression":"zlib"},
            "q3":{"sig_digits":3, "compression":"zlib"},
            "q5":{"sig_digits":5, "compression":"zlib"},
            "r1":{"round_digits":1, "compression":"zlib"},
            "r3":{"round_digits":3, "compression":"zlib"},
            "r5":{"round_digits":5, "compression":"zlib"},
            }


    '''
    """
    Iterate over configurations and record file sizes and a subset of the data
    """
    sizes = []
    for name,args in run_args.items():
        ## Write a netCDF file with the given configuration
        out_path = out_dir.joinpath(f"{dataset_name}_{name}.nc")
        write_to_nc(
                name=dataset_name,
                labels=labels,
                coords=coords,
                data=data,
                out_path=out_path,
                **args
                )
        ## Re-load the data
        new_labels,new_coords,new_data = load_from_nc(dataset_name, out_path)
        ## Extract outgoing fluxes wrt wavelength at the surface
        ## ['idatm', 'zcloud', 'tcloud', 'nre', 'sza', 'wl', 'z', 'feats']
        tmp_data = new_data[0,0,0,0,0,:,-1,:].astype(np.float64)
        file_size = out_path.stat().st_size
        sizes.append((name, file_size))
        ## Remove the netCDF just created
        out_path.unlink()
        ## Save the extracted subset as an npy file
        np.save(out_dir.joinpath(out_path.stem+"_sub.npy"), tmp_data)
    ## Save the file size results for all runs in a JSON
    json.dump(sizes, sizes_path.open("w"))
    '''

    '''
    """ Make a bar plot showing the file sizes after each compression run """
    labels,sizes = zip(*json.load(sizes_path.open("r")),)
    cratios = np.amax(sizes)/np.array(sizes)
    plot_sizes(
            results=dict(zip(labels,sizes)),
            byte_scale=1e9,
            plot_spec = {
                "title":"netCDF4 sizes given different compression schemes",
                "xlabel":"Data storage scheme",
                "ylabel":"File size (GB)",
                }
            )
    '''

    """ Load the array subsets extracted from the compressed files """
    subset = list(sorted([
            (p.stem.split("_")[1],np.load(p))
            for p in out_dir.iterdir()
            if "_sub" in p.stem
            ], key=lambda t:t[0]))
    sub_labels,sub_arrays = zip(*subset)
    ## Stack the arrays from all the runs into (run,wavelength,feature) shape
    sub_arr = np.stack(sub_arrays)

    feature_idx = -1 ## Outgoing flux
    wls = coords[-3]
    wl_cutoff = 120 ## 2.4-4um is outside solar curve so ~0

    '''
    """ Plot spectral curves with each compression method. """
    fig,ax = plt.subplots()
    for i in range(sub_arr.shape[0]-1):
        ax.plot(np.abs(wls[:wl_cutoff]), sub_arr[i,:wl_cutoff,feature_idx],
                label=sub_labels[i])
    ax.set_xlabel("Wavelength ($\mu m$)")
    ax.set_ylabel("Outgoing Flux ($W\,m^{-2}\,um^{-1}$)")
    ax.set_title("Outgoing spectral flux at surface after data reduction")
    ax.legend()
    plt.show()
    '''

    #'''
    """ Calculate and plot error magnitudes """
    fig,ax = plt.subplots()
    init_values = sub_arr[1,:wl_cutoff,feature_idx]
    diffs = np.stack([sub_arr[i,:wl_cutoff,feature_idx]-init_values
                      for i in range(sub_arr.shape[0])])
    print(sub_labels)
    print(f"error: {list(np.average(np.abs(diffs),axis=-1))}")
    error_count = np.array(diffs==0.0).astype(int)
    error_rate = 1-np.sum(error_count,axis=1)/diffs.shape[1]
    print(f"imperfect rate: {error_rate}")
    for i in range(sub_arr.shape[0]-1):
        ax.scatter(wls[:wl_cutoff], diffs[i], label=sub_labels[i])
    ax.set_yscale("log")
    ax.set_ylim([1e-14,1])
    ax.set_xlabel("Wavelength ($\mu m$)")
    ax.set_ylabel("Flux error ($W\,m^{-2}\,um^{-1}$)")
    ax.set_title("Floating point error in flux after data reduction")
    ax.legend()
    plt.show()
    #'''
