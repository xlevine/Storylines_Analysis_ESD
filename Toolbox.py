import gc
import re
import numpy as np
import pickle

from scipy import stats

from os import listdir, remove
from os.path import isfile, join, exists
from netCDF4 import Dataset
from math import pi, cos, sin


from Set_Grid import compute_lonlat_bnds,unigrid,produce_bounds_from_axis,unigrid_var,compute_unigrid_area,spatial_average,regional_cropping,regional_outline,selfdef

institution_model = {'AS-RCEC':['TaiESM1'], 'AWI':['AWI-CM-1-1-MR'], 'BCC':['BCC-CSM2-MR'], 'CAMS':['CAMS-CSM1-0'], 'CAS':['CAS-ESM2-0','FGOALS-f3-L','FGOALS-g3'], 'CCCR-IITM':['IITM-ESM'], 'CCCma':['CanESM5','CanESM5-1','CanESM5-CanOE'], 'CMCC':['CMCC-CM2-SR5','CMCC-ESM2'], 'CNRM-CERFACS':['CNRM-CM6-1','CNRM-ESM2-1'], 'CSIRO':['ACCESS-ESM1-5'], 'CSIRO-ARCCSS':['ACCESS-CM2'], 'E3SM-Project':['E3SM-1-0','E3SM-1-1','E3SM-1-1-ECA'], 'EC-Earth-Consortium':['EC-Earth3','EC-Earth3-AerChem','EC-Earth3-CC','EC-Earth3-Veg','EC-Earth3-Veg-LR'], 'FIO-QLNM':['FIO-ESM-2-0'], 'HAMMOZ-Consortium':['MPI-ESM-1-2-HAM'], 'INM':['INM-CM4-8','INM-CM5-0'], 'IPSL':['IPSL-CM6A-LR'], 'KIOST':['KIOST-ESM'], 'MIROC':['MIROC-ES2L','MIROC6'], 'MOHC':['HadGEM3-GC31-LL','HadGEM3-GC31-MM', 'UKESM1-0-LL'], 'MPI-M':['MPI-ESM1-2-HR','MPI-ESM1-2-LR'], 'MRI':['MRI-ESM2-0'], 'NASA-GISS':['GISS-E2-1-G','GISS-E2-2-G','GISS-E2-1-H'], 'NCAR':['CESM2','CESM2-WACCM','CESM2-FV2'], 'NCC':['NorESM2-LM','NorESM2-MM'], 'NIMS-KMA':['KACE-1-0-G'], 'NOAA-GFDL':['GFDL-CM4','GFDL-ESM4'], 'NUIST':['NESM3'], 'THU':['CIESM'], 'UA':['MCM-UA-1-0']}

models_=list(institution_model.values())
model_list=[model for model_nested in models_ for model in model_nested]

years_expt={'historical':['1985','2014'], 'ssp585':['2070','2099'], 'ssp370':['2070','2099'], 'hist1960':['1960','1989'], 'historical-10yrs':['2005','2014'], 'ssp585-10yrs':['2090','2099'], 'ERA5':['1985','2014']}
scenario_expt={'historical':'historical', 'ssp585':'ssp585', 'ssp370':'ssp370', 'hist1960':'historical', 'historical-10yrs':'historical', 'ssp585-10yrs':'ssp585', 'ERA5':'ERA5'}
table_mon_var={'Amon':['clivi', 'clt', 'clwvi', 'evspsbl', 'hfls', 'hfss', 'hurs', 'hus', 'huss', 'pr', 'prsn', 'prw', 'psl', 'rlds', 'rlus', 'rlut', 'rsds', 'rsdt', 'rsus', 'rsut', 'tauu', 'tauv', 'sfcWind', 'ta', 'tas', 'tasmax', 'tasmin', 'ts', 'ua', 'va', 'zg'], 'Omon':['mlotst', 'sos', 'tos', 'wfo', 'zos', 'zostoga', 'uo_g', 'vo_g'], 'SImon':['sisnconc', 'sisnmass', 'sisnthick', 'sithick', 'siu', 'siv', 'sivol', 'siconc','siconca']}

month_to_index={'jan':[1], 'feb':[2], 'mar':[3], 'apr':[4], 'may':[5], 'jun':[6], 'jul':[7], 'aug':[8], 'sep':[9], 'oct':[10], 'nov':[11], 'dec':[12], 'winter':[11, 12, 1, 2, 3, 4], 'summer':[5, 6, 7, 8, 9, 10], 'annual':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'DJFM':[12, 1, 2, 3], 'JJAS':[6, 7, 8, 9]}


#'BK_seas':[70, 90, 0, 120]
#'Arctic':[50, 90, 0, 360] 

from Set_Grid import delgrid,radius,Omega,gv

## resolution of uniform grid on which all CMIP6 model are regridded upon
#delgrid=0.5
## Earth constants
#radius = 6370000.0
#Omega = 7.2921e-5
#gv = 9.8
 
# dummy variables: some functions require dummy target variables (for evaluation of the predictors only) 
dummy_varname='tas'; dummy_season='annual'; dummy_region='Global'

# Author: Xavier J. Levine (NORCE-AS)
# read_cmip6_clim_model_season: store multi-year, seasonal-mean climatology for all members and models [M x m x D] 
# read_cmip6_monthly_series: load monthly series of variable [D x t], for a given model & member
# read_cmip6_clim: compute multi-year, seasonal-mean climatology [D], for a given model & member 
# read_cmip6_clim_members: load multi-year, seasonal-mean climatology for all members [m x D], for a given model
######################## LOAD CMIP6 
def read_cmip6_clim_model_season(expt, varname, season):

    [var_2D_model, axis, model_list] = read_cmip6_clim_model(expt, varname, season)

    return var_2D_model, axis, model_list

def read_cmip6_clim_model_all(expt, month):

    var_list = ['sos']
#    var_list = ['ua_850hPa']
#    var_list = ['tos','siconc','pr','tas'] 
#    var_list = ['ua_850hPa','ta_850hPa','ua_250hPa']
#    var_list = ['tas']
#    var_list = ['ua_850hPa','ua_250hPa']
#    var_list = ['zos','zostoga','tauuo','tauvo'] 
#    var_list = ['siconc']
    for var in var_list:
        read_cmip6_clim_model(expt, var, month, True)

def compute_gradient_x(var_2D,x_2D):

    x_2D=np.squeeze(x_2D); var_2D=np.squeeze(var_2D)
    Ny=np.shape(x_2D)[0]
    dx_var_2D=np.nan*np.zeros(np.shape(var_2D))
    for ny in np.arange(0,Ny,1):
        dx_var_2D[ny,:] = np.gradient(var_2D[ny,:],x_2D[ny,:])

    return dx_var_2D

def compute_gradient_y(var_2D,y_2D):

    y_2D=np.squeeze(y_2D); var_2D=np.squeeze(var_2D)
    Nx=np.shape(y_2D)[1]
    dy_var_2D=np.nan*np.zeros(np.shape(var_2D))
    for nx in np.arange(0,Nx,1):
        dy_var_2D[:,nx] = np.gradient(var_2D[:,nx],y_2D[:,nx])
    
    return dy_var_2D

def read_cmip6_clim_model(expt, var, month, compute=False):

    # institution_model
    scenario = scenario_expt[expt]
    years = years_expt[expt]

    models_=list(institution_model.values())
    model_list=[model for model_nested in models_ for model in model_nested]
    print(model_list)
#    model_list = ['GISS-E2-1-G', 'GISS-E2-2-G', 'GISS-E2-1-H'] #, 'KACE-1-0-G', 'NESM3', 'MCM-UA-1-0'
    if expt=='ERA5':
        model_list = ['ERA5']
        filename = expt + '_' + var + '_' + month + '_ERA5'
    else:
        filename = expt + '_' + var + '_' + month + '_members'

    if compute is False:
        with open(filename+".pkl", "rb") as f:
            var_model_season_unigrid = pickle.load(f)

        with open("axis_unigrid.pkl", "rb") as f:
            axis_unigrid = pickle.load(f)

    else:
        var_model_season_unigrid={}
        for model in model_list:
            print(model)
            try:
                var_model_season_unigrid[model]={}
                [var_members_season, axis, members_list] = read_cmip6_clim_members(scenario, model, var, years, month)
                for member in members_list:
                    print(member)
                    try: 
                        [var_members_season_unigrid, axis_unigrid] = unigrid_var(np.asarray(var_members_season[member]), axis, delgrid)
                        var_model_season_unigrid[model][member] = var_members_season_unigrid
                    except:
                        print('no data loaded for ', member, ' in ', model)
            except:
               print('no data loaded for ', model) 
               pass

        # save matrix
        with open(filename+'.pkl', "wb") as f:
            pickle.dump(var_model_season_unigrid, f)
            f.close()

        axis_unigrid=unigrid(delgrid)
        with open("axis_unigrid.pkl", "wb") as f:
            pickle.dump(axis_unigrid, f)
            f.close()


    models_list=list(var_model_season_unigrid.keys())

    return var_model_season_unigrid, axis_unigrid, models_list

def read_cmip6_clim_members(scenario, model, var, years, month):

    if model=='ERA5':
        common_members=['r1i1p1f1']
    else:
        common_members = find_members(scenario,model,var)

    var_members_season={}; members_list=[]
    for member in common_members:
        try:            
            [var_data_season, axis]=read_cmip6_clim(scenario, model, member, var, years, month)
            var_members_season[member]=var_data_season
            members_list.append(member)
        except:
            print('1: data not found for member: ' + str(member))

    return var_members_season, axis, members_list

def read_cmip6_clim(expt, model, member, var, years, month):

    try:
        if model=='ERA5':
            var_data, axis = read_era5_monthly_series(var, years)            
        else:
            var_data, axis = read_cmip6_monthly_series(scenario_expt[expt], model, member, var, years)
    except:
        print('0: data not found')
        return

    var_monthly=[]
    for k in np.arange(0,12,1):
        var_monthly_ = np.nanmean(var_data[k:-1:12,],axis=0)
        var_monthly.append(var_monthly_)

    var_data_monthly=np.asarray(var_monthly)
    var_data_season=get_season_mean(var_data_monthly,month)

    return var_data_season, axis

def read_era5_monthly_series(varname, years):

    dic_cmip6_to_era5={'tas':'t2m', 'psl':'msl', 'tos':'sst', 'siconc':'siconc', 'ps':'sp', 'pr':'tp', 'ua':'u', 'ta':'t'}

    date_start='1984'+'01'
    date_end='2015'+'12'
    start_date=years[0]+'01'
    end_date=years[-1]+'12'    
    if 'hPa' in varname:
        varname_split=varname.split('_')[0]
        var=dic_cmip6_to_era5[varname_split]
    else:
        var=dic_cmip6_to_era5[varname]

    data_path_era5="/cluster/projects/nn8002k/ERA5/"

    if ('ta_' in varname) or ('ua_' in varname):
        nc_file_era5="ERA5_ua_ta.nc"
    else:
        nc_file_era5="ERA5_tos_siconc_tas_pr_psl_ps.nc"

    var_file = Dataset(data_path_era5+nc_file_era5, 'r')
    var_ = var_file.variables[var]
    miss_val = var_file.variables[var].missing_value

    if ('ta_' in varname) or ('ua_' in varname):
        plev = np.array(var_file.variables['level'][:])
        plev_eval = int(re.findall(r'\d+', varname)[0])
        k_lev = np.amax(np.argwhere(plev==plev_eval))
        var_data = np.squeeze(var_[:,k_lev,:,:])
    else:
        var_data = np.asarray(var_)

    lon_ = var_file.variables['longitude']
    lon = np.asarray(lon_)
    lat_ = var_file.variables['latitude']    
    lat = np.asarray(lat_)
    var_file.close()

    num_year_start=int(years[0])-int(date_start[:4]);t_start=num_year_start*12
    num_year_end=int(date_end[:4])-int(date_start[:4]);t_end=(num_year_end+1)*12
    var_data_timeseries = var_data[t_start:t_end,:,:]
    k_r = np.where(var_data_timeseries==miss_val)
    var_data_timeseries[k_r]=np.nan

    lon_2D=np.matlib.repmat(lon,len(lat),1)
    lat_2D=np.matlib.repmat(lat,len(lon),1).T   

    [lon_bnds,lat_bnds] = compute_lonlat_bnds(lon,lat)
    
    if varname=='tos':
        var_data_timeseries=var_data_timeseries-273.15
    if varname=='siconc':
        var_data_timeseries=var_data_timeseries*100.0
    if varname=='pr':
        convert_to_mmday = 86400.0/997.0*1.e3
        var_data_timeseries=var_data_timeseries*1000.0/convert_to_mmday

    axis={'lon':lon_2D, 'lon_bnds':lon_bnds, 'lat':lat_2D, 'lat_bnds':lat_bnds}

    return var_data_timeseries, axis

def read_cmip6_monthly_series(expt, model, member, varname, years):

    if 'hPa' in varname:
        var=varname.split('_')[0]
    else:
        var=varname

    [ncfiles, data_path, table_id, var] = find_data_path(expt, model, member, var)

    start_date=years[0]+'01'
    end_date=years[-1]+'12'    

    # read dates for each file 
    fdate_start=[]; fdate_end=[]
    for f in ncfiles:
        fdate=re.search("([0-9]{6}\-[0-9]{6})", f)[0]
        fdate_start.append(fdate[0:6])
        fdate_end.append(fdate[-6:])

    nc_date_start=sorted(fdate_start)
    nc_date_end=[x for _, x in sorted(zip(fdate_start, fdate_end))]
    nc_files=[x for _, x in sorted(zip(fdate_start, ncfiles))]

    # check continuity of files (to be completed)
    out_flag = check_date_continuity(nc_date_start,nc_date_end)
    if out_flag is False:
        print('missing files for ' + var + ' in ' + member + ' ' + model )
        return

    # Select files contained by dates of interest
    if len(nc_date_start)==1:
        index_start=0; date_start=nc_date_start[index_start]
        index_end=0; date_end=nc_date_end[index_end]
    else:

        if (start_date>nc_date_end[-1]) or (end_date>nc_date_end[-1]):
            print('ERROR: time frame exceeds available period for ' + var + ' in ' + member + ' ' + model)
            return
        if (start_date<nc_date_start[0]) or (end_date<nc_date_start[0]):
            print('ERROR: time frame preceeds available period for' + var + ' in ' + member + ' ' + model)
            return

        for i in range(len(nc_files)):
            if (nc_date_start[i]<=start_date) and (start_date<=nc_date_end[i]):
                index_start=i
            if (nc_date_start[i]<=end_date) and (end_date<=nc_date_end[i]):
                index_end=i
        date_start=nc_date_start[index_start]
        date_end=nc_date_end[index_end]

    # open each file and append data to a matrix
    var_data=[]
    for i in np.arange(index_start,index_end+1):
        var_file = Dataset(data_path+nc_files[i], 'r')
        var_ = var_file.variables[var]
        miss_val = var_file.variables[var].missing_value
        if len(np.shape(var_))>3:
            try:
                plev = np.array(var_file.variables['plev'][:])
                plev_eval = int(re.findall(r'\d+', varname)[0])*100
                k_lev = np.amax(np.argwhere(plev==plev_eval))
                var_k = np.squeeze(var_[:,k_lev,:,:])
                del var_; gc.collect()
            except:
                print("level " + str(plev_eval) + "Pa not found for " + var + " in " + member + ' ' + model +"!" )
                return
        else:
            var_k = np.asarray(var_)
            del var_; gc.collect()
        var_data.extend(var_k)
        del var_k; gc.collect()

    var_file.close()

    var_data=np.asarray(var_data)    
    num_year_start=int(years[0])-int(date_start[:4]);t_start=num_year_start*12
    num_year_end=int(date_end[:4])-int(date_start[:4]);t_end=(num_year_end+1)*12
    var_data_timeseries = var_data[t_start:t_end,:,:]
    del var_data; gc.collect()
    k_r = np.where(var_data_timeseries==miss_val)
    var_data_timeseries[k_r]=np.nan
    del k_r; gc.collect()

    if (table_id=='SImon') or (table_id=='Omon'):
        members = find_members(expt,model,'tos'); member=members[0]
        [ncfiles_, data_path, _, _] = find_data_path(expt, model, member, 'tos')
        var_path = data_path + ncfiles_[0]
        [lon,lon_bnds,lat,lat_bnds] =  read_coordinates(var_path,True)
    else:
        [ncfiles_, data_path, _, _] = find_data_path(expt, model, member, var)        
        var_path = data_path + ncfiles_[0]
        [lon,lon_bnds,lat,lat_bnds] = read_coordinates(var_path)

    if np.shape(var_data_timeseries)[1]!=np.shape(lat)[0]:
        print('LAT Omon dim != LAT SImon dim')
        nlat=np.shape(var_data_timeseries)[1]
        lat=lat[:nlat,:]; lon=lon[:nlat,:]#; lat_bnds=lat_bnds[:nlat,:,:]; lon_bnds=lon_bnds[:nlat,:,:]
        if len(np.shape(lat_bnds))==3:            
            lat_bnds=lat_bnds[:nlat,:,:]
        if len(np.shape(lon_bnds))==3:
            lon_bnds=lon_bnds[:nlat,:,:]            

    if np.shape(var_data_timeseries)[2]!=np.shape(lat)[1]:
        print('LON Omon dim != LON SImon dim')
        nlon=np.shape(var_data_timeseries)[2]        
        lat=lat[:,:nlon]; lon=lon[:,:nlon]#; lat_bnds=lat_bnds[:,:nlon,:]; lon_bnds=lon_bnds[:,:nlon,:]
        if len(np.shape(lat_bnds))==3:
            lat_bnds=lat_bnds[:,:nlon,:]
        if len(np.shape(lon_bnds))==3:
            lon_bnds=lon_bnds[:,:nlon,:]            


    # data quality
    if varname=='tos':
        if np.nanmax(var_data_timeseries)>=100.0:
            var_data_timeseries=var_data_timeseries-273.15
    if varname=='siconc':
        if np.nanmax(var_data_timeseries)<=2.0:
            var_data_timeseries=var_data_timeseries*100.0

    axis={'lon':lon, 'lon_bnds':lon_bnds, 'lat':lat, 'lat_bnds':lat_bnds}

    return var_data_timeseries, axis

##########################################################################

def get_data_root(expt, model):

    if expt=='historical':
#        root_path='/trd-project1/NS9001K/ESGF/CMIP6/CMIP/'
        root_path='/cluster/work/users/xale/ESGF/CMIP6/CMIP/'
#        root_path='/cluster/projects/nn8002k/xle041/BootCamp_Arctic/CMIP6/CMIP/'
    else:
#        root_path='/trd-project4/NS8002K/ESGF/CMIP6/ScenarioMIP/'
        root_path='/cluster/work/users/xale/ESGF/CMIP6/ScenarioMIP/'
#        root_path='/cluster/projects/nn8002k/xle041/BootCamp_Arctic/CMIP6/ScenarioMIP/'
    institution_ = [k for k, v in institution_model.items() if model in v]
    institution="".join(institution_)
    expt_path=institution+"/"+model+"/"+expt+"/"

    return root_path, expt_path

def find_data_path(expt, model, member, var):
    
    [root_path, expt_path]=get_data_root(expt, model)

    table_id_ = [k for k, v in table_mon_var.items() if var in v]
    table_id="".join(table_id_)
    
    var_path=member+"/"+table_id+"/"+var+"/"

    if exists(root_path+expt_path+var_path)==False:
        if var=='siconc':
            var='siconca'
            var_path=member+"/"+table_id+"/"+'siconca'+"/"
            if exists(root_path+expt_path+var_path)==False:
                print(var + " not found in " + model)
                return
        else:
            print(var + " not found in " + model)
            return

    grid_all=listdir(root_path+expt_path+var_path)
    if (var in table_mon_var['Omon']) or (var in table_mon_var['SImon']):
        if 'gn' in grid_all:
            grid='gn'
        else:
            grid=grid_all[0]
    else:
        grid=grid_all[0]
    
    version=sorted(listdir(root_path+expt_path+var_path+grid+"/"))[-1]
    data_path=root_path+expt_path+var_path+grid+"/"+version+"/"

    # read file names
    ncfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    return ncfiles, data_path, table_id, var

def check_date_continuity(start_date_list, end_date_list):
    
    year_list_s=[]; month_list_s=[]
    for f in start_date_list:
        year_list_s.append(f[0:4])
        month_list_s.append(f[4:6])
    year_list_e=[]; month_list_e=[]
    for f in end_date_list:
        year_list_e.append(f[0:4])
        month_list_e.append(f[4:6])

    out_flag=True
    for i in np.arange(1,len(start_date_list),1):
        if int(year_list_s[i])!=int(year_list_e[i-1])+1:
            out_flag=False
            break
        if int(month_list_s[i])==1:
            if int(month_list_e[i-1])!=12:
                out_flag=False
                break
        elif int(month_list_s[i])!=int(month_list_e[i-1])+1:
            out_flag=False
            break

    return out_flag

def find_common_members(expt1,expt2,model,var):

    member1=find_members(expt1,model,var)
    member2=find_members(expt2,model,var)    
    common_members=sorted(list(set(member1).intersection(member2)))

    return common_members

def find_members(expt,model,var):

    scenario=scenario_expt[expt]
    years=years_expt[expt]

    [root_path, expt_path]=get_data_root(scenario, model)
    member_list=listdir(root_path+expt_path)
    member=sorted(member_list)
    
    return member

def read_coordinates(var_path,opt_latlon_bnds=False):

    var_file = Dataset(var_path, 'r')

    try:
        lon_ = var_file.variables['lon']
        lat_ = var_file.variables['lat']
    except KeyError:
        pass

    try:
        lon_ = var_file.variables['nav_lon']
        lat_ = var_file.variables['nav_lat']        
    except KeyError:
        pass

    try:
        lon_ = var_file.variables['longitude']
        lat_ = var_file.variables['latitude']
    except KeyError:
        pass

    lon_ = np.asarray(lon_); lat = np.asarray(lat_) 
    ####
    if len(np.shape(lon_))==1:
        lon=np.matlib.repmat(lon_,len(lat_),1)
    else:
        lon=selfdef(lon_)
    if len(np.shape(lat_))==1:
        lat=np.matlib.repmat(lat_,len(lon_),1).T   
    else:
        lat=selfdef(lat_)
    ####

    lon_bnds=[];lat_bnds=[]
    try:
        lon_bnds = var_file.variables['lon_bnds'][:]
        lat_bnds = var_file.variables['lat_bnds'][:]
    except KeyError:
        pass
        
    try:
        lon_bnds = var_file.variables['vertices_longitude'][:]
        lat_bnds = var_file.variables['vertices_latitude'][:]
    except KeyError:
        pass

    try:
        lon_bnds = var_file.variables['bounds_nav_lon'][:]
        lat_bnds = var_file.variables['bounds_nav_lat'][:]
    except KeyError:
        pass

    try:
        lon_bnds = var_file.variables['bounds_lon'][:]
        lat_bnds = var_file.variables['bounds_lat'][:]
    except KeyError:
        pass
    
    if len(lon_bnds)==0:
        print('lat/lon bnds are not computed!')

        # if using atmospheric variable, we MUST define bounds for conservative regridding
        if opt_latlon_bnds==False:
            [lon_bnds,lat_bnds] = compute_lonlat_bnds(lon,lat)
        # if using oceanic / sea-ice variable, bounds are not necessary since bilinear regridding is used
        else:
            lon_bnds=[]; lat_bnds=[]

    var_file.close()

    return lon,lon_bnds,lat,lat_bnds

#####################################
def get_season_mean(var_data_monthly,month):

    k_array=month_to_index[month]
    k_array=list(np.asarray(k_array)-1)
    var_data_season=np.nanmean(var_data_monthly[k_array,:],axis=0)

    return var_data_season

###### TOOLS
######################################

def operator_for_perturbation(function,var_2D_model_member_ctrl,var_2D_model_member_pert):

    models_list_ctrl=list(var_2D_model_member_ctrl.keys())
    models_list_pert=list(var_2D_model_member_pert.keys())

    if 'ERA5' in models_list_ctrl:
        c_models_list=models_list_pert
    else:
        c_models_list=list(set(models_list_ctrl).intersection(models_list_pert))

    common_models_list = []
    for model in models_list_pert:
        if model in c_models_list:
            common_models_list.append(model)

    if 'ERA5' in models_list_ctrl:
        var_2D_common_model_member_ctrl = var_2D_model_member_ctrl        
    else:
        var_2D_common_model_member_ctrl = extract_models_dic(var_2D_model_member_ctrl,common_models_list)
    var_2D_common_model_member_pert = extract_models_dic(var_2D_model_member_pert,common_models_list)

    output_common_model_member={} 
    for model in common_models_list:
        output_common_model_member[model] = {}
        members_pert=list(var_2D_common_model_member_pert[model].keys()) 
        if 'ERA5' in models_list_ctrl:
            members_list=members_pert
        else:
            members_ctrl=list(var_2D_common_model_member_ctrl[model].keys()) 
            members_list=list(set(members_ctrl).intersection(members_pert))
        for member in members_list:
            var_2D_common_model_pert = np.asarray(var_2D_common_model_member_pert[model][member])
            if 'ERA5' in models_list_ctrl:
                var_2D_common_model_ctrl = np.asarray(var_2D_common_model_member_ctrl['ERA5']['r1i1p1f1'])
            else:
                var_2D_common_model_ctrl = np.asarray(var_2D_common_model_member_ctrl[model][member])
            output_common_model_member[model][member]= function(var_2D_common_model_ctrl,var_2D_common_model_pert)

    return output_common_model_member

def operator_for_members(function, var_members_model, *args):

    models_list=list(var_members_model.keys())
    output_members_model={}
    for model in models_list:
        var_members_model_=var_members_model[model]
        members=list(var_members_model_.keys())
        output_members_model[model]={}
        for member in members:
#            var_members=np.asarray(var_members_model[model][member])
            var_members=np.squeeze(var_members_model[model][member])
            output_members_model[model][member]=function(var_members,args)            

    return output_members_model

def remove_outliers_dic(var_dic,weight_area,axis,region):

    # compute global spatial average, max, min values
    lon=axis['lon']; lat=axis['lat']
    var_1D_dic=operator_for_members(spatial_average, var_dic, weight_area, lon, lat, region, 'avg')

    var_1D_val=list(NestedDictValues(var_1D_dic))
    [var_lower_bound,var_upper_bound] = compute_quartile(var_1D_val)
    var_dic_out=exclude_models_with_nans(var_dic,var_1D_dic,var_lower_bound,var_upper_bound)

    return var_dic_out

def exclude_models_with_nans(var_dic,var_1D_dic,var_lower_bound,var_upper_bound):

    models_list=list(var_1D_dic.keys())
    var_dic_out={}
    a=0; b=0
    for model in models_list:
        var_dic_out[model]={}
        members_list=list(var_1D_dic[model].keys())
        for member in members_list:
            if (var_1D_dic[model][member]<=var_upper_bound) and (var_1D_dic[model][member]>=var_lower_bound):
                var_dic_out[model][member]=var_dic[model][member]
                a+=1
            else:
                b+=1
                print(model + ' - ' + member + ' excluded')
      
    print(str(b) + " of " + str(a+b) + " simulations excluded")

    return var_dic_out

def extract_models_dic(var_dic,models_list):

    var_key_dic={}
    for model in models_list:
        var_data = var_dic[model]
        var_key_dic[model] = var_data

    return var_key_dic

def NestedDictValues(d):

    for v in d.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v

def compute_models_ensmean_dic(var_dic):
    
    models_list = list(var_dic.keys())
    var_ensmean_dic={}
    for model in models_list:
        var_members=var_dic[model]
        try:
            members_list=list(var_members.keys())
            array_members=[]
            for member in members_list:
                array_members.append(np.asarray(var_members[member]))
            mean_members=np.nanmean(array_members,axis=0)
            var_ensmean_dic[model]=mean_members
        except:
            var_ensmean_dic[model]=var_members

    return var_ensmean_dic

def convert_pred_dic_to_ensmean_array(predvars_region_model,models_list):

    predictor_list=list(predvars_region_model.keys())
    predvars_region_ensmean_array=[]
    for predictor in predictor_list:
        predvar_region_model_array=[]
        for model in models_list:
            predvar_region_member_array=[]
            try:
                member_list=list(predvars_region_model[predictor][model].keys())
                predvar_region_model_member=[]
                for member in member_list:
                    predvar_region_model_member.append(predvars_region_model[predictor][model][member])
                predvar_region_model_=np.nanmean(predvar_region_model_member)
                predvar_region_model_array.append(predvar_region_model_)
            except:
                predvar_region_model_=predvars_region_model[predictor][model]
                predvar_region_model_array.append(predvar_region_model_)
        predvars_region_ensmean_array.append(predvar_region_model_array)

    predvars_region_ensmean_array=np.asarray(predvars_region_ensmean_array)

    return predvars_region_ensmean_array

def convert_dic_to_array_2D(data_dic):

    data_array=[]; models_out_list=[]
    models_list=list(data_dic.keys())
    for model in models_list:
        data_=np.asarray(data_dic[model])
        if len(np.shape(data_))!=0:
            data_array.append(data_)
            models_out_list.append(model)
    data_array = np.asarray(data_array)

    return data_array, models_out_list

def convert_dic_to_array(data_dic):

    data_array=[];
    models_list = list(data_dic.keys())
    for model in models_list:
        data_=np.asarray(data_dic[model])
        data_array.append(data_)
    data_array = np.asarray(data_array)

    return data_array

def convert_num_to_array(num,*args):

    array=[]
    array.append(num)
    array=np.asarray(array)

    return array

def find_common_models(targetvar_2D_model,predvars_region_model):

    # select only models that are common between variables    
    targetvar_2D_model_list=list(targetvar_2D_model.keys())
    common_model_list_s=[]
    predictors_input_list=list(predvars_region_model.keys())
    for predname in predictors_input_list: 
        common_model_list_=list(predvars_region_model[predname].keys())
        if not common_model_list_s:
            common_model_list_s=common_model_list_
        common_model_list_s=list(set(common_model_list_s).intersection(common_model_list_))
    common_model_list_s=list(set(common_model_list_s).intersection(targetvar_2D_model_list))
    
    # re-order model 
    common_model_list=[]
    for model in targetvar_2D_model_list:
        if model in common_model_list_s:
            common_model_list.append(model)

    # find index of existing models for each list
    targetvar_2D_common_model = extract_models_dic(targetvar_2D_model,common_model_list)

    predvars_region_common_model={}
    for predname in predictors_input_list: 
        predvar_region_model = extract_models_dic(predvars_region_model[predname],common_model_list)
        predvars_region_common_model[predname]=predvar_region_model

    return targetvar_2D_common_model, predvars_region_common_model

###############

# remove MMM and normalized by STD
def rm_MMM_var(predvar_region_model_deltaclim):
    
    predvar_region_model_deltaclim_mmm=compute_models_ensmean_dic(predvar_region_model_deltaclim)
    predvar_region_model_deltaclim_array=convert_dic_to_array(predvar_region_model_deltaclim_mmm)

    predvar_region_model_deltaclim_array = convert_dic_to_array(predvar_region_model_deltaclim_mmm)
#    predvar_region_model_deltaclim_array[np.where(np.isinf(predvar_region_model_deltaclim_array))]=np.nan

    predvar_region_model_deltaclim_MMM = np.nanmean(predvar_region_model_deltaclim_array,axis=0)#,keepdims=True)
    predvar_region_model_deltaclim = operator_for_members(subtract_function,predvar_region_model_deltaclim,predvar_region_model_deltaclim_MMM)
    predvar_region_model_deltaclim = operator_for_members(multiply_function,predvar_region_model_deltaclim,-1.0)

    return predvar_region_model_deltaclim

def dv_STD_var(predvar_region_model_deltaclim):

    predvar_region_model_deltaclim_mmm=compute_models_ensmean_dic(predvar_region_model_deltaclim)
    predvar_region_model_deltaclim_array=convert_dic_to_array(predvar_region_model_deltaclim_mmm)
#    predvar_region_model_deltaclim_array[np.where(np.isinf(predvar_region_model_deltaclim_array))]=np.nan
    predvar_region_model_deltaclim_STD=np.nanstd(predvar_region_model_deltaclim_array,axis=0)#,keepdims=True)

    predvar_region_model_deltaclim = operator_for_members(multiply_function,predvar_region_model_deltaclim,1.0/predvar_region_model_deltaclim_STD)

    return predvar_region_model_deltaclim

# normalize by Global-mean surface temperature change
def normalize_var(var_model_deltaclim,normvar_region_model_deltaclim):

    var_2D_common_model_normdeltaclim = operator_for_perturbation(divide_function,normvar_region_model_deltaclim,var_model_deltaclim)

    return var_2D_common_model_normdeltaclim

###############################################

def add_function(X_ctrl,X_pert):

    delta_X = np.add(X_pert,X_ctrl)

    return delta_X

def subtract_function(X_ctrl,X_pert):

    delta_X = np.subtract(X_pert,X_ctrl)

    return delta_X

def divide_function(X_ctrl,X_pert):

    delta_X = np.divide(X_pert,X_ctrl)

    return delta_X

def multiply_function(X_ctrl,X_pert):

    delta_X = np.multiply(X_pert,X_ctrl)

    return delta_X

#######################################

def compute_quartile(data_1D,*args):

    q1, q3= np.nanpercentile(data_1D,[25,75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    return lower_bound,upper_bound

def return_stat_ML(reg, X, y):

    sst = np.sum((np.mean(y,axis=0,keepdims=True) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
    sse = np.sum((reg.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
    se = np.array([np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X)))) for i in range(sse.shape[0])])

    r2 = 1.0 - np.divide(sse,sst)
    t = reg.coef_ / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), y.shape[0] - X.shape[1]))

    return p, r2
