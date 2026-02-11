#!/usr/bin/env python
import os
import numpy as np
import xarray as xr
import scipy.signal as signal
import pandas as pd

varname="tsi"
filter_list=[16*12,18*12,25*12]
N  = 6    # Filter order
infile="/Users/stergios/lpool/indices/tsi/cmip6_tsi_extendedwith_tsiv03r00from2021_1850-2023_mm.nc"

print("read %s"%(infile))
ds = xr.open_dataset(infile,decode_times=False)[varname]
#ds['time']=xr.cftime_range("1850", freq="MS", periods=len(ds.time))
ds['time']=xr.cftime_range("1850", freq="MS", periods=len(ds.time))
var  = np.array(ds.values)
time = ds["time"]

nsteps=np.size(np.array(time))

for sfilter in filter_list:    
   ofile="/Users/stergios/lpool/indices/tsi/cmip6_tsi_extendedwith_tsiv03r00from2021_1850-2023_fltbwPAD%d_mm.nc"%(sfilter)
   print(f'write {ofile}')
 
   npads=int(3*round(sfilter/2.))
   npads0=int(round(sfilter/2.))
   var_pad=np.zeros((nsteps+2*npads))
   var_pad[npads:(npads+nsteps)]=var
   var_pad[0:npads]=np.mean(var[0:npads0])
   var_pad[(npads+nsteps):(nsteps+2*npads)]=np.mean(var[(nsteps-npads0-1):nsteps])
 
   # First, design the Buterworth filter
   Wn=1./(0.5*sfilter) #analyse annual means
 
   b, a = signal.butter(N, Wn, btype='high', analog=False)
   output_pad = signal.filtfilt(b, a, var_pad, axis=0)

   output_hp=output_pad[npads:(npads+nsteps)]
   # calculate lfrequency
   output_lp=var-output_hp
   del output_pad
   del b,a
   ds_o = xr.Dataset({
      'tsi': xr.DataArray(
                data   = var,   # enter data here
                dims   = ['time'],
                coords = {'time': ds['time']},
                attrs  = {
                    '_FillValue': -999.9,
                    'units'     : 'W/m2',
                     'long_name': 'Total Solar Irradiance'
                    }
                ),
      'tsi_hp': xr.DataArray(
                data   = output_hp,   # enter data here
                dims   = ['time'],
                coords = {'time': ds['time']},
                attrs  = {
                    '_FillValue': -999.9,
                    'units'     : 'W/m2',
                    'long_name': f'Total Solar Irradiance high-passed {sfilter} butterworth'
                    }
                ),
      'tsi_lp': xr.DataArray(
                data   = output_lp,   # enter data here
                dims   = ['time'],
                coords = {'time': ds['time']},
                attrs  = {
                    '_FillValue': -999.9,
                    'units'     : 'W/m2',
                    'long_name': f'Total Solar Irradiance low-passed {sfilter} butterworth'
                    }
                )
            },
        attrs = {'Institution': 'Academy of Athens','Author':'Stergios Misios'}
   )
   ds_o.to_netcdf(ofile)
   del output_hp, output_lp
