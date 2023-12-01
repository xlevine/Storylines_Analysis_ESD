import numpy as np
import xesmf as xe
import xarray as xr

from math import pi, cos, sin

domain_latlon={'Global':[-90, 90, 0, 360], 'Tropics':[-30, 30, 0, 360], 'Arctic':[55, 90, 0, 360], 'Antarctic':[-90, -60, 0, 360], 'NH':[0, 90, 0, 360], 'SH':[-90, 0, 0, 360], 'NH_midpolar':[20, 90, 0, 360], 'SH_midpolar':[-90, -20, 0, 360], 'NH_vortex':[70, 80, 0, 360], 'Europe':[30, 65, -20, 50], 'N_Atlantic':[45, 65, -90, 30], 'Scandinavia':[55, 75, 0, 50], 'NH_MidPole':[40, 90, 0, 360], 'NH_midlatitude':[45, 60, 0, 360], 'BK_seas':[65, 80, 26, 95], 'NE_Atl':[45, 60, -70, -30], 'NW_Atl':[45, 65, -30, 20], 'Beaufort':[72.5, 80, 180, 225], 'Central_Arctic':[70, 90, 0, 360], 'GrnlandSeas':[50, 75, -75, -15], 'High_Arctic':[65, 90, 0, 360], 'North_Atl':[45, 60, -70, 0], 'North_Pac':[45, 60, 130, 240], 'NPac_Midlat':[35, 60, 130, 240], 'NH_Midlat':[35, 60, 0, 360], 'North_Hemisphere':[0, 60, 0, 360]}

# resolution of uniform grid on which all CMIP6 model are regridded upon
delgrid=0.5

# Earth constants
radius = 6370000.0
Omega = 7.2921e-5
gv = 9.8

##############################
def compute_lonlat_bnds(lon,lat):

    # lon_bnds
    if (max(lon)<180) or (max(lon)>360):
        print('longitude incorrect format')
        return
    else:
        lon_bds=np.empty(len(lon)+1)
        if (min(lon)>0) and (max(lon)<360):
            lon_bds[0]=0; lon_bds[-1]=360.0
            lon_bds[1:-1]=(lon[0:-1]+lon[1:])/2
        elif ((min(lon)==0) or (max(lon)==360.0)):
            lon_bds[0]=(lon[-1]-360.0+lon[0])/2
            lon_bds[-1]=(lon[0]+360.0+lon[-1])/2
            lon_bds[1:-1]=(lon[0:-1]+lon[1:])/2

    # lat_bnds
    lat_bds=np.empty(len(lat)+1)
    lat_bds[0]=-90.0; lat_bds[-1]=90.0
    lat_bds[1:-1]=(lat[0:-1]+lat[1:])/2

    lon_bnds=[]
    for i in np.arange(0,len(lon),1):
        lon_bnds.append([lon_bds[i],lon_bds[i+1]])
    lat_bnds=[]
    for i in np.arange(0,len(lat),1):
        lat_bnds.append([lat_bds[i],lat_bds[i+1]])

    lon_bnds=np.asarray(lon_bnds)
    lat_bnds=np.asarray(lat_bnds)    

    return lon_bnds, lat_bnds

def unigrid(delgrid):

    lon_b_out=np.arange(0, 360+delgrid, delgrid)
    lat_b_out=np.arange(-90, 90+delgrid, delgrid)
    lon_out = 0.5*(lon_b_out[1:]+lon_b_out[:-1])
    lat_out = 0.5*(lat_b_out[1:]+lat_b_out[:-1])
    axis_unigrid={'lon':lon_out, 'lon_bnds':lon_b_out, 'lat':lat_out, 'lat_bnds':lat_b_out}

    return axis_unigrid

def produce_bounds_from_axis(axis):

    lon_in=axis['lon']; lat=axis['lat']

    return lon_bnds_in, lat_bnds_in

def unigrid_var(var_data, axis_in, delgrid):

    # output grid
    axis_unigrid=unigrid(delgrid)
    lon_out=axis_unigrid['lon']; lat_out=axis_unigrid['lat']
    lon_b_out=axis_unigrid['lon_bnds']; lat_b_out=axis_unigrid['lat_bnds']
    grid_out_xr = xr.Dataset()
    grid_out_xr.coords["lon"] = (("x"),lon_out)
    grid_out_xr.coords["lat"] = (("y"),lat_out)
    grid_out_xr.coords["lon_b"] = (("x_b"),lon_b_out)
    grid_out_xr.coords["lat_b"] = (("y_b"),lat_b_out)

    # input grid
    lon_in_=axis_in['lon']
    lat_in_=axis_in['lat']

    if len(np.shape(lon_in_))==1:
        lon_in=np.matlib.repmat(lon_in_,len(lat_in_),1)
    else:
        lon_in=selfdef(lon_in_)

    if len(np.shape(lat_in_))==1:
        lat_in=np.matlib.repmat(lat_in_,len(lon_in_),1).T
    else:
        lat_in=selfdef(lat_in_)

    regrid_method='bilinear'    
    k_r = np.where(np.isnan(var_data))
    var_data[k_r]=0
    land_mask = np.ones(np.shape(var_data))
    land_mask[k_r]=0
#    lon_in[k_r]=np.nan
#    lat_in[k_r]=np.nan

    grid_in_xr = xr.Dataset()
    grid_in_xr.coords["lon"] = (("y", "x"),lon_in)
    grid_in_xr.coords["lat"] = (("y", "x"),lat_in)

    regridder = xe.Regridder(grid_in_xr, grid_out_xr, regrid_method, ignore_degenerate=True, periodic=True)
    var_data_unigrid = regridder(var_data)
    
    if len(np.shape(lon_in))==2:
        land_mask_unigrid = regridder(land_mask)
        var_data_unigrid=np.divide(var_data_unigrid,land_mask_unigrid)

    return var_data_unigrid, axis_unigrid

def compute_unigrid_area(lat):

#    lat=axis['lat'];
    lat_rad=lat*np.pi/180
    delgrid=np.amax(np.diff(lat))
    delgrid_rad=delgrid*np.pi/180    
    delta_lat=delgrid_rad
    delta_lon=delgrid_rad
        
    dim_lon=int(360./delgrid)
    dim_lat=int(180./delgrid)

    lat_2d_T = np.matlib.repmat(lat_rad,dim_lon,1)
    lat_2d = lat_2d_T.T
    
    area_2d = radius**2*delta_lon*delta_lat * np.cos(lat_2d)

    return area_2d

def spatial_average(var_data,*args):

    tuple_input=np.squeeze(args); weight_area=tuple_input[0]; lon=tuple_input[1]; lat=tuple_input[2]; region=tuple_input[3]; avg_flag=tuple_input[4] 

    if len(np.shape(var_data))==3:
        area=np.tile(weight_area, (np.shape(var_data)[0], 1, 1))
    elif (len(np.shape(var_data))==1) and (len(np.shape(weight_area))==2):
        var_data=np.tile(var_data, (np.shape(weight_area)[1],1)).T
        area=selfdef(weight_area)
    elif len(np.shape(var_data))==2:
        area=selfdef(weight_area)
    else:
        print('array dimension incorrect!')

    area[np.where(np.isnan(var_data))]=0; area[np.where(var_data==0)]=0
    var_data_area=np.multiply(var_data,area)

    # lat in 0 dim; lon is regular 0-360 
    lon_flag=any(n < 0 for n in lon)
    if lon_flag==True:
        print('ERROR: longitude is not in standard form')
        return

    [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
    var_area_region = regional_cropping(var_data_area,bound_box)
    area_region = regional_cropping(area,bound_box)
    if len(np.shape(var_data))==2:
        var_area_regsum=np.nansum(var_area_region)
        area_regsum=np.nansum(area_region)
    elif len(np.shape(var_data))==3:
        var_area_regsum=np.nansum(var_area_region,axis=(1,2))
        area_regsum=np.nansum(area_region,axis=(1,2))
    else:
        print('ERROR: data array is not in standard form')
        return

    if avg_flag=='avg':
        var_area_regsum=var_area_regsum/area_regsum
        
    if not np.shape(var_area_regsum):
        if var_area_regsum==0:
            var_area_regsum=np.nan
    else:
        var_area_regsum[np.where(var_area_regsum==0)]=np.nan

    return var_area_regsum

def regional_cropping(var_data,*args):

    tuple_input=np.squeeze(args); bound_box=tuple_input
    ind_lon_w=bound_box[0]; ind_lon_e=bound_box[1]; ind_lat_s=bound_box[2]; ind_lat_n=bound_box[3];

    if len(np.shape(var_data))==2:
        var_data_wrap=np.concatenate((var_data,var_data),axis=1) 
        var_data_region=var_data_wrap[ind_lat_s:ind_lat_n+1,ind_lon_w:ind_lon_e+1]
    elif len(np.shape(var_data))==3:
        var_data_wrap=np.concatenate((var_data,var_data),axis=2) 
        var_data_region=var_data_wrap[:,ind_lat_s:ind_lat_n+1,ind_lon_w:ind_lon_e+1]
    else:
        print('ERROR: data array is not in standard form')
        return

    return var_data_region

def regional_outline(lat,lon,region): 

    lon_wrap=np.concatenate((lon-360,lon))
    lat_s=domain_latlon[region][0]; lat_n=domain_latlon[region][1]
    lon_w=domain_latlon[region][2]; lon_e=domain_latlon[region][3]

    ind_lat_s=np.amin(np.argwhere(lat_s<=lat)); ind_lat_n=np.amax(np.argwhere(lat_n>=lat))
    ind_lon_w=np.amin(np.argwhere(lon_w<=lon_wrap)); ind_lon_e=np.amax(np.argwhere(lon_e>=lon_wrap))
    lat_region=lat[ind_lat_s:ind_lat_n+1]; lon_region=lon_wrap[ind_lon_w:ind_lon_e+1]

    bound_box = [ind_lon_w, ind_lon_e, ind_lat_s, ind_lat_n]

    return lat_region, lon_region, bound_box

#######################################

def selfdef(varin):

   varout = [];
   varout.append(varin)
   varout = np.squeeze(np.asarray(varout))

   return varout
