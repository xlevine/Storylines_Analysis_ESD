import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import math

import numpy as np
from numpy import inf
import numpy.ma as ma

import csv
import pickle

import pandas as pd
from pandas.plotting import table 


from scipy import stats
from scipy.ndimage.filters import gaussian_filter

from Toolbox import read_cmip6_clim_model_season, remove_outliers_dic, compute_models_ensmean_dic, convert_dic_to_array_2D, regional_outline, regional_cropping, operator_for_perturbation, subtract_function, compute_unigrid_area, NestedDictValues, extract_models_dic, operator_for_members, spatial_average, rm_MMM_var, convert_dic_to_array, convert_pred_dic_to_ensmean_array, read_cmip6_clim_model
from empirical_clustering import return_eof_modes, return_mca_modes

from Toolbox import table_mon_var, institution_model, dummy_varname, dummy_season, dummy_region
from Master import predictors_input_list, prepare_ML_input, compute_MultiLinear, compute_storylines, create_landocean, return_ensmean_from_members, read_cmip6_deltaclim_model, read_cmip6_normdeltaclim_model, return_clim_ensmean_model, compute_eof_storylines
from Toolbox import model_list
from Set_Grid import domain_latlon, selfdef

domain_project={'nh_polar':['Arctic', 'NH_midpolar', 'NH_vortex', 'NH_midlatitude','NH_MidPole','Central_Arctic', 'High_Arctic'], 'sh_polar':['Antarctic', 'SH_midpolar'], 'plane':['Global', 'Tropics', 'NH', 'SH', 'Europe', 'N_Atlantic', 'Scandinavia']}

# Author: Xavier J. Levine (NORCE-AS)
# print_target: plot model / multi-model climatology or response to climate change
# print_pred_plot: plot parameter space for predictors
# print_response: plot target response to predictors
# print_pred_boxplot: plot box plot for predictor changes
# print_response_R2: plot variance explained by Multivariate Linear Regression
# print_storylines: plot a specific storyline

def print_all(expt_ctrl, expt_pert, tar_name):

    tar_season='summer'

    tar_region='Arctic'
#    tar_region='NH_MidPole'

    print_pred_boxplot(expt_ctrl, expt_pert)
    print_pred_plot(expt_ctrl, expt_pert)
    print_response(expt_ctrl, expt_pert, tar_name, tar_season, tar_region)
    print_response_R2(expt_ctrl, expt_pert, tar_name, tar_season, tar_region)
    print_storylines(expt_ctrl, expt_pert, tar_name, tar_season, tar_region)
    print_storylines(expt_ctrl, expt_pert, tar_name, tar_season, tar_region, 'EOF')
    print_MMM_change(expt_pert, tar_name, tar_season, tar_region, expt_ctrl, norm=True)

def print_MMM_change(expt, varname, season, region, expt_ctrl, norm=True, norm_varname='tas', norm_season='annual', norm_region='Global'):


    # load change in climatology
    if norm==True:
        [var_2D_model_dic, axis, models_list] = read_cmip6_normdeltaclim_model(expt_ctrl, expt, varname, season, norm_varname, norm_season, norm_region)
    else:
        [var_2D_model_dic, axis, models_list] = read_cmip6_deltaclim_model(expt_ctrl, expt, varname, season)

    lon=axis['lon']; lat=axis['lat']; area_unigrid = compute_unigrid_area(lat)

    var_2D_model_dic = remove_outliers_dic(var_2D_model_dic,area_unigrid,axis,'NH_midpolar')
    var_2D_model_ensmean_dic = compute_models_ensmean_dic(var_2D_model_dic)

    # load present-day climatology
    [var_clim_2D_model_dic, axis, clim_models_list] = read_cmip6_clim_model_season(expt_ctrl, varname, season)
    var_clim_2D_model_dic = remove_outliers_dic(var_clim_2D_model_dic,area_unigrid,axis,'NH_midpolar')
    var_clim_2D_model_ensmean_dic = compute_models_ensmean_dic(var_clim_2D_model_dic)

    # remove bad models
    models_to_removed=['FGOALS-f3-L','FGOALS-g3']
    for model in models_to_removed:
        try:
            del var_2D_model_ensmean_dic[model]
            del var_clim_2D_model_ensmean_dic[model]
        except:
            print('no data for ', model)

    # Change in climatology
    [var_2D_model, models_out_list]=convert_dic_to_array_2D(var_2D_model_ensmean_dic)
    var_2D_mean = np.nanmean(var_2D_model,axis=0)
    [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
    var_2D_region = regional_cropping(var_2D_mean, bound_box)

    # present-day climatology
    [var_clim_2D_model, clim_models_out_list]=convert_dic_to_array_2D(var_clim_2D_model_ensmean_dic)
    var_clim_2D_mean = np.nanmean(var_clim_2D_model,axis=0)
    [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
    var_clim_2D_region = regional_cropping(var_clim_2D_mean, bound_box)

    # find areas where 80% of models agree
    var_2D_model_sign_p = selfdef(var_2D_model)
    var_2D_model_sign_m = selfdef(var_2D_model)
    var_2D_model_sign_p[np.where(var_2D_model>0)]=1
    var_2D_model_sign_p[np.where(var_2D_model<=0)]=0
    var_2D_model_sign_m[np.where(var_2D_model<0)]=1
    var_2D_model_sign_m[np.where(var_2D_model>=0)]=0
    var_2D_model_sign_p_mean=np.nanmean(var_2D_model_sign_p,axis=0)
    var_2D_model_sign_m_mean=np.nanmean(var_2D_model_sign_m,axis=0)
    var_2D_agreement=np.ones(np.shape(var_2D_model_sign_p_mean))
    var_2D_agreement[np.where(var_2D_model_sign_p_mean>0.8)]=0
    var_2D_agreement[np.where(var_2D_model_sign_m_mean>0.8)]=0
    var_2D_sign = regional_cropping(var_2D_agreement, bound_box) 
    
    table_id_ = [k for k, v in table_mon_var.items() if varname in v]
    table_id="".join(table_id_)
    if (table_id=='Omon') or (table_id=='SImon'):
        mask_2D = create_landocean('Ocean',region)
        var_2D_region=np.multiply(mask_2D,var_2D_region)
    elif (table_id=='Lmon') or (table_id=='LImon'):
        mask_2D = create_landocean('Land',region)
        var_2D_region=np.multiply(mask_2D,var_2D_region)

    if varname=='pr':
        cmap='PuOr'
        convert_to_mmday = 86400.0/997.0*1.e3
        var_2D_region = var_2D_region * convert_to_mmday
        var_clim_2D_region = var_clim_2D_region * convert_to_mmday
    elif varname=='psl':
        cmap='RdBu_r'
        var_2D_region = var_2D_region * 0.01
        var_clim_2D_region = var_clim_2D_region * 0.01
#    elif varname=='tas':
#        cmap='autumn'
    else:
        cmap='RdBu_r'

    # plot data
#    title_name = varname + '_' + season + '_' + expt + '_delta_' + expt_ctrl + '_' + model + '_' + region
#    title_name = varname + '_' + season + '_' + expt + '_delta' #_' + expt_ctrl # + '_' + model + '_' + region
    title_name = r'$\Delta$' + varname + ':MMM'
    var_2D_max = np.nanmax([np.nanmax(var_2D_region),-np.nanmin(var_2D_region)])

    if varname=='pr':
        var_2D_max = 0.3
        min_val_n=0; max_val_n=0
        min_val_p=0; max_val_p=8.0
    if varname=='ua_850hPa':
        var_2D_max = 0.3
        min_val_n=-10.0; max_val_n=3.0;
        min_val_p=3.0; max_val_p=10.0;
    if varname=='tas':
        var_2D_max=2.1
        min_val_n=0; max_val_n=0
        min_val_p=260.0; max_val_p=300.0;
    if varname=='siconc':
        var_2D_max=14.0
        min_val_n=0; max_val_n=0
        min_val_p=15.0; max_val_p=100.0

    nval_p = (max_val_p-min_val_p)/7.0
    nval_n = (max_val_n-min_val_n)/7.0
    if nval_p!=0:
        levels_p = np.arange(min_val_p, max_val_p, nval_p)
    else:
        levels_p = [0]
    if nval_n!=0:
        levels_n = np.arange(min_val_n, max_val_n, nval_n)
    else:
        levels_n = [0]
    
    if (varname=='tas'):
#            color_ticks=np.arange(0,var_2D_max*(1+1./20),(var_2D_max)/20)
        color_ticks=np.arange(-var_2D_max,var_2D_max*(1+1./10),(2*var_2D_max)/10)                
    else:
        color_ticks=np.arange(-var_2D_max,var_2D_max*(1+1./10),(2*var_2D_max)/10)                

    plot_Response(var_2D_region,var_2D_sign,lon_region,lat_region,color_ticks,title_name,region,cmap,var_clim_2D_region,levels_n,levels_p)

def print_target(expt, model, varname, season, region, expt_ctrl='dummy', norm=True, norm_varname='tas', norm_season='annual', norm_region='Global'):
    
    [var_2D_region, lon_region, lat_region] = return_clim_ensmean_model(expt, model, varname, season, region, expt_ctrl, norm, norm_varname, norm_season, norm_region)

    var_2D_region=np.squeeze(var_2D_region)
 
    # RM MMM
#    var_2D_region = var_2D_region - np.nanmean(var_2D_region,axis=0,keepdims=True)
    
    table_id_ = [k for k, v in table_mon_var.items() if varname in v]
    table_id="".join(table_id_)
    if (table_id=='Omon') or (table_id=='SImon'):
        mask_2D = create_landocean('Ocean',region)
        var_2D_region=np.multiply(mask_2D,var_2D_region)
    elif (table_id=='Lmon') or (table_id=='LImon'):
        mask_2D = create_landocean('Land',region)
        var_2D_region=np.multiply(mask_2D,var_2D_region)

    if varname=='pr':
        cmap='PuOr'
        convert_to_mmday = 86400.0/997.0*1.e3
        var_2D_region = var_2D_region * convert_to_mmday
    elif varname=='psl':
        cmap='RdBu_r'
        var_2D_region = var_2D_region * 0.01
    else:
        cmap='RdBu_r'

    # plot data
    if varname=='sisnthick':
        var_2D_region[np.where(var_2D_region>1.0e2)]=np.nan

    if expt_ctrl!='dummy':
        title_name = varname + '_' + season + '_' + expt + '_delta_' + expt_ctrl + '_' + model + '_' + region
#        title_name = varname + '_' + season + '_' + expt + '_delta' #_' + expt_ctrl # + '_' + model + '_' + region
        var_2D_max = np.nanmax([np.nanmax(var_2D_region),-np.nanmin(var_2D_region)])

        if varname=='pr':
           var_2D_max = 0.3
        if varname=='ua_850hPa':
           var_2D_max = 0.5
        if varname=='zos':
#            var_2D_max=0.1
#            var_2D_max=0.2
            var_2D_max=0.15
#            var_2D_max = np.nanmax([np.nanmax(var_2D_region),-np.nanmin(var_2D_region)])
#            if var_2D_max<0.125:
#                var_2D_max=0.1
#            else:
#                var_2D_max=0.2
        if varname=='psl':
            var_2D_max=1.5
        if (varname=='uo_g') or (varname=='vo_g'):
#            var_2D_max=0.012
            var_2D_max=0.008
        if varname=='tas':
    #           cmap='Reds'
            var_2D_max=2.4
        if varname=='ta_850hPa':
    #           cmap='Reds'
            var_2D_max=2.0

        if 'EOF' in model:
            var_2D_max = np.nanmax(var_2D_region); var_2D_min = np.nanmin(var_2D_region)
            color_ticks=np.arange(-var_2D_max,var_2D_max*(1+1./20),(2*var_2D_max)/20)
        elif (varname=='tas') or (varname=='ta_850hPa'):
            color_ticks=np.arange(0,var_2D_max*(1+1./20),(var_2D_max)/20)
#        elif (varname=='siconc'):
#            color_ticks=np.arange(-var_2D_max,0,(var_2D_max)/20)
        else:
            color_ticks=np.arange(-var_2D_max,var_2D_max*(1+1./20),(2*var_2D_max)/20)                

        if expt_ctrl=='ERA5':
            if varname=='tos':
                var_2D_max=2.5
            if varname=='siconc':
                var_2D_max=20.0
            if varname=='tas':
                var_2D_max=3.0
            if 'ta_' in varname:
                var_2D_max=2.0
            if 'ua_' in varname:
                var_2D_max=2.0
            if varname=='pr':
                var_2D_max=0.8
            color_ticks=np.arange(-var_2D_max,var_2D_max*(1+1./20),(2*var_2D_max)/20)                

    else:
        title_name=varname + '_' + season + '_' + expt + '_' + model + '_' + region
#        var_2D_region[np.where(np.absolute(var_2D_region)==1.e20)]=np.nan
        var_2D_max = np.nanmax(var_2D_region); var_2D_min = np.nanmin(var_2D_region)
        if varname=='siconc':
            var_2D_max=100; var_2D_min=0
        if varname=='sisnthick':
            var_2D_max=0.3; var_2D_min=0
        if varname=='zos':
            var_2D_max=1.2; var_2D_min=-1.2
        if varname=='psl':
            var_2D_max=1025.0; var_2D_min=1000.0
        if varname=='tos':
            var_2D_max=12.0; var_2D_min=0.0
        if varname=='pr':
#            var_2D_max=7.0; var_2D_min=0.0
            var_2D_max=0.7; var_2D_min=0.0
        if (varname=='uo_g') or (varname=='vo_g'):
#            var_2D_max=0.12; var_2D_min=-0.12
            var_2D_max=0.024; var_2D_min=-0.024

        if 'EOF' in model:
            color_ticks=np.arange(-var_2D_max,var_2D_max*(1+1./20),(2*var_2D_max)/20)
            title_name=varname + '_' + expt + '_' + model
        else:
            color_ticks=np.arange(var_2D_min,var_2D_max+(var_2D_max-var_2D_min)/10.0,(var_2D_max-var_2D_min)/10.0)

    plot_data(var_2D_region,lon_region,lat_region,color_ticks,title_name,region,cmap)


def print_pred_plot(expt_ctrl, expt_pert, predictors_input=predictors_input_list, rm_MMM=True, norm_GW=True, scaled=True):

    # NOT scaled by global warming index
    [targetvar_2D_common_model_deltaclim, predvars_region_common_model_deltaclim, axis_region] = prepare_ML_input(expt_ctrl, expt_pert, dummy_varname, dummy_season, dummy_region, predictors_input, rm_MMM, norm_GW, scaled)

    common_models_list=list(targetvar_2D_common_model_deltaclim.keys())

    already_done=[]
    for i in np.arange(0,len(predictors_input['predictor']),1):
        already_done.append(predictors_input['predictor'][i])
        for j in np.arange(0,len(predictors_input['predictor']),1):
            if (i!=j) and (predictors_input['predictor'][j] not in already_done):                
                x=predvars_region_common_model_deltaclim[predictors_input['predictor'][i]]       
                y=predvars_region_common_model_deltaclim[predictors_input['predictor'][j]] 
                if scaled==True:
                    if predictors_input['predictor'][i]!='GWI':
#                        xlabel='scaled_'+predictors_input['predictor'][i] XL
                        xlabel=predictors_input['predictor'][i]
                    else:
                        xlabel=predictors_input['predictor'][i]
                        x=rm_MMM_var(x)

                    if predictors_input['predictor'][j]!='GWI':
#                        ylabel='scaled_'+predictors_input['predictor'][j] XL
                        ylabel=predictors_input['predictor'][j]
                    else:
                        ylabel=predictors_input['predictor'][j]
                        y=rm_MMM_var(y)

                    xlim=[-3,3]; ylim=[-3,3]

                else:
                    xlabel=predictors_input['predictor'][i]
                    ylabel=predictors_input['predictor'][j]

                    y_values = np.asarray(list(NestedDictValues(y)))
                    x_values = np.asarray(list(NestedDictValues(x)))

                    ymin = np.nanmin(y_values)
                    ymax = np.nanmax(y_values)
                    #1.25*np.max([np.absolute(np.nanmin(y_values)),np.absolute(np.nanmax(y_values))])
                    xmin = np.nanmin(x_values)
                    xmax = np.nanmax(x_values) 
                    #1.25*np.max([np.absolute(np.nanmin(x_values)),np.absolute(np.nanmax(x_values))])
                    xlim = [xmin, xmax]; ylim = [ymin, ymax]
                    
#                title =ylabel+'_vs_'+xlabel+'_'+expt_pert+'_delta_'+expt_ctrl

                title =ylabel+'_vs_'+xlabel
                plot_model_scatterplot(x,y,common_models_list,xlabel,ylabel,xlim,ylim,title)

def print_response(expt_ctrl, expt_pert, tar_name, tar_season, tar_region, predictors_input=predictors_input_list):

    norm_varname=predictors_input['varname'][predictors_input['predictor'].index('GWI')]
    norm_season=predictors_input['season'][predictors_input['predictor'].index('GWI')]
    norm_region=predictors_input['region'][predictors_input['predictor'].index('GWI')]

    # Perform multilinear regression
    [ML_coeffs, ML_intercept, Pred_std, p_value, Score, axis_region, X, models_list] = compute_MultiLinear(expt_ctrl, expt_pert, tar_name, tar_season, tar_region, predictors_input)

    # Load MMM climatology
    [Mean, lon_region, lat_region] = return_clim_ensmean_model(expt_pert, 'MMM', tar_name, tar_season, tar_region, expt_ctrl, True, norm_varname, norm_season, norm_region)

    # compute coefficient associated with predictor
    for i in np.arange(0,len(predictors_input['predictor'])-1,1):
        pred_name = predictors_input['predictor'][i+1]
        ML_coeff = np.squeeze(ML_coeffs[i,:,:])
        Coefficient = ML_coeff * Pred_std[i]
        Pval = np.squeeze(p_value[i,:,:])
             
#        title_name = 'coefficient_' + tar_name + '_to_' + pred_name + '_' + expt_pert + '_delta_' + expt_ctrl
#        title_name = tar_name + '_to_' + pred_name
        title_name = r'$\Delta$' + tar_name + ' from ' + pred_name

        if tar_name=='pr':
            cmap='PuOr'
            convert_to_mmday = 86400.0/997.0*1.e3
            Coefficient = Coefficient * convert_to_mmday
            var_2D_max = 0.07
        elif tar_name=='ua_850hPa':
            cmap='RdBu_r'
            var_2D_max = 0.08
        elif tar_name=='tas':
            cmap='RdBu_r'
            var_2D_max = 0.29
        elif (tar_name=='uo_g') or (tar_name=='vo_g'):
            cmap='RdBu_r'
#            var_2D_max=0.012
            var_2D_max=0.008
        elif tar_name=='zos':
            cmap='RdBu_r'
            var_2D_max = 0.06
        else:
            cmap='RdBu_r'
            var_2D_max = np.nanmax([np.nanmax(Coefficient),-np.nanmin(Coefficient)])
            
        max_val = np.nanmax(Mean); min_val = np.nanmin(Mean)
        nval =  (max_val-min_val)/6
        if min_val>=0:        
            levels_p = np.arange(min_val, max_val, nval)
            levels_n = np.arange(-max_val, -min_val, nval)
        else:
            levels_p = np.arange(nval, max_val, nval)
            levels_n = np.arange(min_val, nval, nval)

        color_ticks = np.arange(-var_2D_max,var_2D_max*(1.+1./10),(2*var_2D_max)/10);
        plot_Response(Coefficient,Pval,lon_region,lat_region,color_ticks,title_name,tar_region,cmap)
#        plot_data(Coefficient,lon_region,lat_region,color_ticks,title_name,tar_region,cmap,Mean,levels_n,levels_p)

def print_pred_boxplot(expt_ctrl, expt_pert, predictors=predictors_input_list, norm_GW=False, rm_MMM=False, scaled_MM=False):

    # NOT scaled by global warming index
    [targetvar_2D_common_model_deltaclim, predvars_region_common_model_deltaclim, axis_region] = prepare_ML_input(expt_ctrl, expt_pert, dummy_varname, dummy_season, dummy_region, predictors, rm_MMM, norm_GW, scaled_MM)

    common_models_list=list(targetvar_2D_common_model_deltaclim.keys())
    predvars_region_ensmean_deltaclim_array = convert_pred_dic_to_ensmean_array(predvars_region_common_model_deltaclim,common_models_list)
    predvars_nans=np.mean(predvars_region_ensmean_deltaclim_array,axis=0)
    predvars_nans_index=np.where(~np.isnan(predvars_nans))

    for i in range(len(predictors['predictor'])):
        if predictors['predictor'][i]=='SIE_Arctic':
            predvars_region_ensmean_deltaclim_array[i,:]=predvars_region_ensmean_deltaclim_array[i,:]/(1.0e14)
        elif predictors['predictor'][i]=='PSL_Beaufort':
            predvars_region_ensmean_deltaclim_array[i,:]=predvars_region_ensmean_deltaclim_array[i,:]/(1.0e2)

    predvars_region_ensmean_deltaclim_array=np.squeeze(predvars_region_ensmean_deltaclim_array[:,predvars_nans_index])

    data_box = predvars_region_ensmean_deltaclim_array.T
    fig = plt.figure()
    ax = plt.axes()
    bp = plt.boxplot(data_box,sym='') #,labels=predictors['predictor'])# +['GWI'])
    ax.set_xticklabels(labels=predictors['predictor'], rotation=45, Fontsize=12)
    plt.ylabel('[K]', Fontsize=12)
    ax.set_ylim([0,11])
#    ax.set_yticklabels(labels=['0','','2','','4','','6','','8','','10',''])
    save_file = 'pred_variance_' + expt_pert + '_delta_' + expt_ctrl + '.png'
    fig.savefig(save_file,bbox_inches='tight')
    plt.close()

def print_response_R2(expt_ctrl, expt_pert, tar_name, tar_season, tar_region, predictors_input=predictors_input_list):

    # Perform multilinear regression
    [ML_coeffs, ML_intercept, Pred_std, p_value, Score, axis_region, X, models_list] = compute_MultiLinear(expt_ctrl, expt_pert, tar_name, tar_season, tar_region, predictors_input)
    lon_region=axis_region['lon']; lat_region=axis_region['lat']

    # plot data
    title_name= 'R2_' + tar_name + '_' + tar_region
    var_2D_max = 0.9; var_2D_min = 0; cmap='YlOrRd'
    color_ticks=np.arange(var_2D_min,var_2D_max+(var_2D_max-var_2D_min)/10,(var_2D_max-var_2D_min)/10)
    plot_data(Score,lon_region,lat_region,color_ticks,title_name,tar_region,cmap)

def print_storylines(expt_ctrl, expt_pert, tar_name, tar_season, tar_region, predictors_input=predictors_input_list):

    # example storyline='Tropical_Ampli+|Polar_Ampli+'
    if 'EOF' in predictors_input:
        [Storyline, axis_region] = compute_eof_storylines(expt_ctrl, expt_pert, tar_name, tar_season, tar_region)
    else:
        [Storyline, axis_region] = compute_storylines(expt_ctrl, expt_pert, tar_name, tar_season, tar_region, predictors_input)
    lon_region=axis_region['lon']; lat_region=axis_region['lat']

    for k in range(len(Storyline['name'])):

        #    title_name = tar_name + '_' + expt_pert + '_delta_' + expt_ctrl + ':' + storyline
#        title_name = tar_name + ':' + Storyline['name'][k]
        title_name = r'$\Delta$' + tar_name + ':' + Storyline['name'][k]
        Storyline_data = Storyline['data'][k]

        cmap='RdBu_r'
        if tar_name=='pr':
            cmap='PuOr'
            convert_to_mmday = 86400.0/997.0*1.e3
            Storyline_data = Storyline_data * convert_to_mmday
#            var_2D_max = 0.3
            var_2D_max = 0.12
        elif tar_name=='ua_850hPa':
#            var_2D_max = 0.5
            var_2D_max = 0.16
        elif tar_name=='tas':
#            cmap='Reds'
#            var_2D_max = 2.4
            var_2D_max = 0.64
        elif tar_name=='siconc':
            var_2D_max = 4.0
        elif (tar_name=='uo_g') or (tar_name=='vo_g'):
#            var_2D_max=0.012
            var_2D_max=0.008
        else:
            var_2D_max = np.nanmax([np.nanmax(Storyline_data),-np.nanmin(Storyline_data)])

        if tar_name=='tas':
#            color_ticks=np.arange(0,var_2D_max*(1+1./20),(var_2D_max)/20)
            color_ticks = np.arange(-var_2D_max,var_2D_max*(1.+1./10),(2*var_2D_max)/10)
        else:
            color_ticks = np.arange(-var_2D_max,var_2D_max*(1.+1./10),(2*var_2D_max)/10)

        plot_data(Storyline_data,lon_region,lat_region,color_ticks,title_name,tar_region,cmap)

########################## TOOLS
def plot_data(var_2d,lon,lat,color_ticks,title_name,region,color_map_name,clim_var_2d='None',contour_n='None',contour_p='None'):

    # determine projection
    for key, value in domain_project.items():
        print(key, value)
        if region in value:
            project_map=key
    
    if project_map=='nh_polar':
        proj=ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
        lat_bnd=np.min(lat)
    elif project_map=='sh_polar':
        proj=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)
        lat_bnd=np.max(lat)
    else:
        proj=ccrs.Robinson()

    fig = plt.figure()
    # set up a map
    ax = plt.axes(projection=proj)
    if project_map.find('polar')>=0:
        if project_map.find('nh')>=0:
            ax.set_extent([-180, 180, lat_bnd, 90], crs=ccrs.PlateCarree())
        elif project_map.find('sh')>=0:
            ax.set_extent([-180, 180, -90, lat_bnd], crs=ccrs.PlateCarree())
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    # even bounds gives a contour-like effect
#    bounds = np.linspace(-1, 1, 11)
    bounds = np.linspace(-1, 1, 21)
#    bounds = np.linspace(-1, 1, len(color_ticks)) #11
    # get one more color than bounds from colormap
    colors = plt.get_cmap(color_map_name)(np.linspace(0,1,len(bounds)+1))
    # create colormap without the outmost colors
    color_map = mcolors.ListedColormap(colors[1:-1])
    # set upper/lower color
    color_map.set_over(colors[-1])
    color_map.set_under(colors[0])

    cs=plt.pcolormesh(lon, lat, var_2d, transform=ccrs.PlateCarree(), cmap=color_map, vmin=min(color_ticks), vmax=max(color_ticks))

    if 'polar' in project_map:
        cbar=plt.colorbar(cs,orientation="horizontal",fraction=0.046,pad=0.02,ticks=color_ticks)
    else:
        cbar=plt.colorbar(cs,orientation="horizontal",pad=0.02,ticks=color_ticks)

    plt.clim(min(color_ticks),max(color_ticks))

    color_ticks_label = []
    nticks = len(color_ticks)
    for n in np.arange(0,nticks,1):
        if (n % 2) == 0: 
            color_ticks_label.append(str(round(color_ticks[n],2)))
        else:
            color_ticks_label.append('') 
    cbar.ax.set_xticklabels(color_ticks_label)

    if isinstance(clim_var_2d, str)==False:
        
        ######## contour plot
        plt.contour(lon, lat, clim_var_2d, levels=contour_p, transform=ccrs.PlateCarree(), linestyles='solid', linewidths=2.0, colors='k')
        plt.contour(lon, lat, clim_var_2d, levels=contour_n, transform=ccrs.PlateCarree(), linestyles='dashed', linewidths=1.0, colors='k')

#    plt.plot([0, 360], [50, 50], linestyle='dashed', linewidth=1.0, color='k')
#    plt.plot([25, 95], [65, 65], linestyle='dashed', linewidth=2.0, color='k')
#    plt.plot([25, 95], [80, 80], linestyle='dashed', linewidth=2.0, color='k')
#    plt.plot([25, 25], [65, 80], linestyle='dashed', linewidth=2.0, color='k')
#    plt.plot([95, 95], [65, 80], linestyle='dashed', linewidth=2.0, color='k')

    # define Region Mask
    lon_2D=np.matlib.repmat(lon,len(lat),1); lat_2D=np.matlib.repmat(lat,len(lon),1).T
    lonlat_region=domain_latlon['BK_seas']; lat_south=lonlat_region[0]; lat_north=lonlat_region[1]; lon_west=lonlat_region[2]; lon_east=lonlat_region[3]
    region_mask=np.ones(np.shape(lon_2D)); region_mask[np.where(lat_2D>=lat_north)]=0; region_mask[np.where(lat_2D<=lat_south)]=0; region_mask[np.where(lon_2D>=lon_east)]=0; region_mask[np.where(lon_2D<=lon_west)]=0
    ocean_mask_2D = create_landocean('Ocean',region); ocean_mask_2D[np.where(np.isnan(ocean_mask_2D)==True)]=0.0; region_mask=np.multiply(ocean_mask_2D,region_mask)
    plt.contour(lon, lat, region_mask, levels=[0,1], transform=ccrs.PlateCarree(), linestyles='dashed', linewidths=3.0, colors='g')
    plt.contour(lon, lat, region_mask, levels=[0,1], transform=ccrs.PlateCarree(), linestyles='dashed', linewidths=1.0, colors='w')

    ax.coastlines()
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.title(title_name, fontsize=12)
    save_file = title_name + '.png'
    fig.savefig(save_file,bbox_inches='tight')
    plt.close()

def plot_model_scatterplot(X,Y,common_model_list,xlabel,ylabel,xlim,ylim,title_name):

    cmap1 = cm.get_cmap('Paired')
    cmap2 = cm.get_cmap('Dark2')
    cmap3 = cm.get_cmap('Set3')
    rgba_list=[]
    
    for i in range(cmap1.N):
        rgba = cmap1(i)
        rgba_list.append(rgba)
    for i in range(cmap2.N):
        rgba = cmap2(i)
        rgba_list.append(rgba)
    for i in range(cmap3.N):
        rgba = cmap3(i)
        rgba_list.append(rgba)

    marker_list = ['D', 's', 'X', '<', '>', '^', 'o']

    ################
    X_ensmean = return_ensmean_from_members(X)
    Y_ensmean = return_ensmean_from_members(Y)
    ###############

    institution_list=list(institution_model.keys())
    fig = plt.figure()
    ax = plt.axes()
    inst_prev={}; n=0; i=0
    for model in common_model_list:
        for inst, mod in institution_model.items():
#            print(model, inst, mod)
            if model in mod:
                k_inst=institution_list.index(inst)
                inst_now=inst
        color_=rgba_list[k_inst-1]
        if inst_now not in inst_prev:
            marker_=marker_list[0]
            i=0
            inst_prev[inst_now]=1
        else:
            i=inst_prev[inst_now]
            marker_=marker_list[i]
            inst_prev[inst_now]=i+1            
        X_members=list(X[model].keys())
        Y_members=list(Y[model].keys())
        common_members = list(set(X_members).intersection(Y_members))
        for member in common_members:
            ax.plot(X[model][member],Y[model][member],color=color_,marker=marker_,linestyle='None',markersize=1)

        if (np.isnan(X_ensmean[model])==False) and  (np.isnan(Y_ensmean[model])==False):
            ax.plot(X_ensmean[model],Y_ensmean[model],color=color_,marker=marker_,linestyle='None',label=model,markersize=6)
            print(model)

#        n+=1
    ax.plot([0,0],ylim,'--k')
    ax.plot(xlim,[0,0],'--k')
    ax.legend(fontsize=5.5,bbox_to_anchor=(1.0, 1.05))

    # produce 2 arrays from ensmean
    X_ensmean_array=convert_dic_to_array(X_ensmean)
    Y_ensmean_array=convert_dic_to_array(Y_ensmean)

    x=ma.masked_invalid(X_ensmean_array); y=ma.masked_invalid(Y_ensmean_array); msk = (~x.mask & ~y.mask)
    ax.text(0.1, 0.95, 'r2='+str(np.round(np.power(np.corrcoef(y[msk],x[msk])[0,1],2),2)), transform=ax.transAxes)


    if xlim==ylim:
        num_predictor = 2
        xi = stats.chi2.ppf(0.8,num_predictor)
        x_m= xi**0.5; dx=(2*x_m)/100
        x_t=np.arange(-x_m,x_m+dx,dx);
        r=np.corrcoef(y[msk],x[msk])[0][1]
        y_t_p=x_t*r+(x_t**2*(r**2-1.0)+(1-r**2)*xi)**0.5; y_t_p[np.isnan(y_t_p)]=x_t[np.isnan(y_t_p)]*r
        y_t_n=x_t*r-(x_t**2*(r**2-1.0)+(1-r**2)*xi)**0.5; y_t_n[np.isnan(y_t_n)]=x_t[np.isnan(y_t_n)]*r
#    # t = np.sqrt(stats.chi2.ppf(0.8,num_predictor)/2)
#    # y_t=(t-x_t**2)**0.5

        x_r = (0.5*(1-r**2)*xi/(1-r))**0.5
        x_l = (0.5*(1-r**2)*xi/(1+r))**0.5

        plt.plot(x_t,y_t_p,'-k')
        plt.plot(x_t,y_t_n,'-k')
        plt.plot(x_r,x_r,'or')
        plt.plot(x_l,-x_l,'ob')
        plt.plot(-x_l,x_l,'ob')
        plt.plot(-x_r,-x_r,'or')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
#    plt.title(title_name)
    save_file = title_name + '.png'
    fig.savefig(save_file,bbox_inches='tight',dpi=300)                                                                                 
    plt.close()

def plot_Response(Response,Pval,lon,lat,color_ticks,title_name,region,color_map_name,Mean='None',levels_n='None',levels_p='None'):

    # define projection for plot
    for key, value in domain_project.items():
        if region in value:
            project_map=key
                   
    if project_map=='nh_polar':
        proj=ccrs.NorthPolarStereo(central_longitude=0.0, globe=None)
        lat_bnd=np.min(lat)
    elif project_map=='sh_polar':
        proj=ccrs.SouthPolarStereo(central_longitude=0.0, globe=None)
        lat_bnd=np.max(lat)
    else:
        proj=ccrs.Robinson()

    fig = plt.figure()

    # set up a map
    ax = plt.axes(projection=proj)
    if project_map.find('polar')>=0:
        if project_map.find('nh')>=0:
            ax.set_extent([-180, 180, lat_bnd, 90], crs=ccrs.PlateCarree())
        elif project_map.find('sh')>=0:
            ax.set_extent([-180, 180, -90, lat_bnd], crs=ccrs.PlateCarree())
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    # even bounds gives a contour-like effect
#    bounds = np.linspace(-1, 1, 11)
    bounds = np.linspace(-1, 1, 21)

    # get one more color than bounds from colormap
    colors = plt.get_cmap(color_map_name)(np.linspace(0,1,len(bounds)+1))
    # create colormap without the outmost colors
    color_map = mcolors.ListedColormap(colors[1:-1])
    # set upper/lower color
    color_map.set_over(colors[-1])
    color_map.set_under(colors[0])

    ######## color plot        
    cs=plt.pcolormesh(lon, lat, Response, transform=ccrs.PlateCarree(), cmap=color_map, vmin=min(color_ticks), vmax=max(color_ticks))

    if 'polar' in project_map:
        cbar=plt.colorbar(cs,orientation="horizontal",fraction=0.046,pad=0.02,ticks=color_ticks)
    else:
        cbar=plt.colorbar(cs,orientation="horizontal",pad=0.02,ticks=color_ticks)

    plt.clim(min(color_ticks),max(color_ticks))

    color_ticks_label = []
    nticks = len(color_ticks)
    for n in np.arange(0,nticks,1):
        if (n % 2) == 0: 
            color_ticks_label.append(str(round(color_ticks[n],2)))
        else:
            color_ticks_label.append('') 
    cbar.ax.set_xticklabels(color_ticks_label)
    cbar.ax.tick_params(labelsize=12)

    # define Region Mask
    lon_2D=np.matlib.repmat(lon,len(lat),1); lat_2D=np.matlib.repmat(lat,len(lon),1).T
    lonlat_region=domain_latlon['BK_seas']; lat_south=lonlat_region[0]; lat_north=lonlat_region[1]; lon_west=lonlat_region[2]; lon_east=lonlat_region[3]
    region_mask=np.ones(np.shape(lon_2D)); region_mask[np.where(lat_2D>=lat_north)]=0; region_mask[np.where(lat_2D<=lat_south)]=0; region_mask[np.where(lon_2D>=lon_east)]=0; region_mask[np.where(lon_2D<=lon_west)]=0
    ocean_mask_2D = create_landocean('Ocean',region); ocean_mask_2D[np.where(np.isnan(ocean_mask_2D)==True)]=0.0; region_mask=np.multiply(ocean_mask_2D,region_mask)
    plt.contour(lon, lat, region_mask, levels=[0,1], transform=ccrs.PlateCarree(), linestyles='dashed', linewidths=3.0, colors='g')
    plt.contour(lon, lat, region_mask, levels=[0,1], transform=ccrs.PlateCarree(), linestyles='dashed', linewidths=1.0, colors='w')

    if isinstance(Mean, str)==False:

    ######## contour plot
        if len(levels_p)>1:
            plt.contour(lon, lat, Mean, levels=levels_p, transform=ccrs.PlateCarree(), linestyles='solid', linewidths=1.5, colors='tab:gray') # , alpha=0)

        if len(levels_n)>1:
            plt.contour(lon, lat, Mean, levels=levels_n, transform=ccrs.PlateCarree(), linestyles='dashed', linewidths=1.5, colors='tab:gray') # , alpha=0)



#        plt.contour(lon, lat, Mean, levels=levels_p, transform=ccrs.PlateCarree(), linestyles='solid', linewidths=2.0, colors='k') # , alpha=0)
#        plt.contour(lon, lat, Mean, levels=levels_n, transform=ccrs.PlateCarree(), linestyles='dashed', linewidths=1.0, colors='k') #, alpha=0)
    
    ####### dash plot 
    levels = [0, 0.05, 1]
    plt.contourf(lon, lat, Pval, levels=levels, transform=ccrs.PlateCarree(), hatches=[".", ""], alpha=0)
    ########

    ax.coastlines()
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.title(title_name,fontsize=12)
    save_file = title_name + '.png'
    fig.savefig(save_file,bbox_inches='tight')                                                                                 
    plt.close()
