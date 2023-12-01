import numpy as np
import numpy.matlib
from sklearn.linear_model import LinearRegression
from scipy import stats

from Toolbox import read_cmip6_clim_model_season, read_cmip6_clim_model, read_cmip6_clim_members, read_cmip6_clim, read_cmip6_monthly_series
from Toolbox import get_data_root, find_data_path, check_date_continuity
from Toolbox import find_common_members, find_members, find_common_models
from Toolbox import read_coordinates, compute_lonlat_bnds, unigrid_var, compute_unigrid_area, regional_cropping, regional_outline
from Toolbox import spatial_average, get_season_mean
from Toolbox import operator_for_perturbation, operator_for_members, remove_outliers_dic, extract_models_dic, NestedDictValues, exclude_models_with_nans, compute_models_ensmean_dic,convert_pred_dic_to_ensmean_array, convert_dic_to_array_2D, convert_num_to_array, convert_dic_to_array
from Toolbox import rm_MMM_var, dv_STD_var, normalize_var
from Toolbox import add_function, subtract_function, divide_function, multiply_function
from Toolbox import compute_quartile, return_stat_ML
from Toolbox import dummy_varname, dummy_season, dummy_region
from empirical_clustering import return_eof_modes, return_mca_modes

# Author: Xavier J. Levine (NORCE-AS)
# prepare_ML_init : load target [m x D] and predictors [P x m x D], for a given experiment.
# prepare_ML_input: (1) call prepare_ML_init for historical and SSP585/SSP370, (2) compute anomalies as difference  between SSP585/SSP370 and historical, (3) compute spatial-mean of predictors over region of interest, (4) normalize target and predictors by the global and annual-mean surface temperature change, (5) remove multi-model mean and normalize by multi-model standard deviation for predictors, (6) only consider models common to target and all predictors.

# compute_MultiLinear: compute response of targets to all predictors using multivariate linear regression analysis.
# compute_storylines: produce storylines of target for any given combination of predictors.

predictors_input_list={'predictor':['GWI','ArcAmp','BKWarm'], 'varname':['tas','ta_850hPa','tos'], 'region':['Global', 'Arctic', 'BK_seas'], 'season':['annual', 'summer', 'summer']}


# dummy variables: some functions require dummy target variables (for evaluation of the predictors only)
dummy_varname='tas'; dummy_season='annual'; dummy_region='Global'

##################################### COMPUTE MULTILINEAR REGRESSION
def compute_storylines(expt_ctrl, expt_pert, tarname, season, region, predictors_input_ML=predictors_input_list):

    # WARNING: This script only works for bivariate storylines, eg Polar_Ampli+|Tropical_Ampli-
#    phases=['-','+','0']
    phases=['-','+']
    prednames=[predictors_input_ML['predictor'][1],predictors_input_ML['predictor'][2]]

    # compute joint distribution
    num_predictor = len(prednames)
    xi = stats.chi2.ppf(0.8,num_predictor)

    [corr_scaled,corr_unscaled]=compute_pred_corrtable(expt_ctrl, expt_pert, predictors_input_ML)
    r=corr_scaled[1][2]
    t_r = (0.5*(1-r**2)*xi/(1-r))**0.5
    t_l = (0.5*(1-r**2)*xi/(1+r))**0.5

    # Perform multilinear regression
    [ML_coeffs, ML_intercept, Pred_std, p_value, Score, axis_region, X, models_list] = compute_MultiLinear(expt_ctrl, expt_pert, tarname, season, region, predictors_input_ML)

    # compute storylines
    ML_coeff1 = np.squeeze(ML_coeffs[0,:,:])
    ML_coeff2 = np.squeeze(ML_coeffs[1,:,:])

    Storyline={'name':[],'data':[]}
    for phase1 in phases:
        if phase1=='0':
            s1=0
        else:
            s1=np.sign(int(phase1+'1'))
        for phase2 in phases:
            if phase2=='0':
                s2=0
            else:
                s2=np.sign(int(phase2+'1'))
                        
            if r>0:
                if s1==s2:
                    t_s=np.max([t_r,t_l])
                else:
                    t_s=np.min([t_r,t_l])
            elif r<0:
                if s1!=s2:
                    t_s=np.max([t_r,t_l])
                else:
                    t_s=np.min([t_r,t_l])

#            Storyline_=ML_intercept+s1*ML_coeff1*t_s+s2*ML_coeff2*t_s 
            Storyline_=s1*ML_coeff1*t_s+s2*ML_coeff2*t_s 
            Storyline['name'].append(prednames[0]+phase1+'|'+prednames[1]+phase2)
            Storyline['data'].append(Storyline_)

    return Storyline, axis_region

# LOAD predictors and target variable in MLR model and obtain response coefficients
def compute_MultiLinear(expt_ctrl, expt_pert, target_varname, target_season, target_region, predictors_input=predictors_input_list):

    [targetvar_2D_common_model_deltaclim, predvars_region_common_model_deltaclim, axis_region] = prepare_ML_input(expt_ctrl, expt_pert, target_varname, target_season, target_region, predictors_input)

    # convert target var to ensemble-mean array
    targetvar_2D_ensmean_deltaclim = compute_models_ensmean_dic(targetvar_2D_common_model_deltaclim)
    [targetvar_2D_ensmean_deltaclim_array,models_out_list] = convert_dic_to_array_2D(targetvar_2D_ensmean_deltaclim)

    # convert pred var to ensemble-mean array
    predvars_region_ensmean_deltaclim_array_=convert_pred_dic_to_ensmean_array(predvars_region_common_model_deltaclim,models_out_list)
    predvars_region_ensmean_deltaclim_array_=np.squeeze(predvars_region_ensmean_deltaclim_array_)
    predvars_region_ensmean_deltaclim_array_[np.where(np.isinf(predvars_region_ensmean_deltaclim_array_))]=np.nan

    predvars_region_ensmean_deltaclim_array=predvars_region_ensmean_deltaclim_array_[1:,:]

    predvars_nans=np.mean(predvars_region_ensmean_deltaclim_array,axis=0)
    predvars_nans_index_=np.where(~np.isnan(predvars_nans))
    predvars_nans_index=predvars_nans_index_[0]
    models_list = [models_out_list[i] for i in predvars_nans_index]
    predvars_region_ensmean_deltaclim_array=np.squeeze(predvars_region_ensmean_deltaclim_array[:,predvars_nans_index])
    targetvar_2D_ensmean_deltaclim_array=np.squeeze(targetvar_2D_ensmean_deltaclim_array[predvars_nans_index,:,:])

    model_dim = np.shape(predvars_region_ensmean_deltaclim_array)[1]
    feature_dim = np.shape(predvars_region_ensmean_deltaclim_array)[0]
    lat_dim = np.shape(targetvar_2D_ensmean_deltaclim_array)[1]
    lon_dim = np.shape(targetvar_2D_ensmean_deltaclim_array)[2]    

    # format input for Linear regression
    y_=targetvar_2D_ensmean_deltaclim_array.reshape((model_dim,lat_dim*lon_dim))
    X=predvars_region_ensmean_deltaclim_array.T
    X_std=np.nanstd(X,axis=0)

    # set NaNs to gridcell if one model has NaN
    common_ind_Val=[]
    for i in np.arange(0,model_dim,1):
        common_ind_Val_ = np.squeeze(np.argwhere(np.isnan(y_[i,:])==False))
        if not common_ind_Val:
            common_ind_Val=common_ind_Val_
        common_ind_Val=list(set(common_ind_Val).intersection(common_ind_Val_))        

    # scikit-learn Linear regression
    y = y_[:,common_ind_Val]
    reg = LinearRegression()
    
    reg.fit(X, y)
    Coeff_ = reg.coef_.T
    Intercept_ = reg.intercept_.T

    Coeff = np.empty((feature_dim,lat_dim*lon_dim)); Coeff[:] = np.NaN
    Coeff[:,common_ind_Val] = Coeff_
    Response_Coeff = Coeff.reshape((feature_dim,lat_dim,lon_dim))

    Intercept = np.empty((lat_dim*lon_dim)); Intercept[:] = np.NaN
    Intercept[common_ind_Val] = Intercept_
    Response_Intercept = Intercept.reshape((lat_dim,lon_dim))
    
    # compute p value and score
    [p_, score_] = return_stat_ML(reg, X, y)

    p = np.empty((feature_dim,lat_dim*lon_dim)); p[:] = np.NaN
    p[:,common_ind_Val] = p_.T
    p_value = p.reshape((feature_dim,lat_dim,lon_dim))

    score = np.empty((lat_dim*lon_dim)); score[:] = np.NaN
    score[common_ind_Val] = score_
    Score = score.reshape((lat_dim,lon_dim))

    return Response_Coeff, Response_Intercept, X_std, p_value, Score, axis_region, X, models_list

# COMPUTE predictors index (1xM), target var in region & season (2DxM) + normalization index of perturbation
def prepare_ML_input(expt_ctrl, expt_pert, target_varname='tas', target_season='annual', target_region='Global', predictors_input=predictors_input_list, rm_MMM=True, norm_GW=True, scaled_MM=True):

    # load predictors and target
    [targetvar_2D_model_hist, predvars_2D_model_hist, axis_global] = prepare_ML_init(expt_ctrl, target_varname, target_season, predictors_input)
    [targetvar_2D_model_ssp, predvars_2D_model_ssp, axis_global] = prepare_ML_init(expt_pert, target_varname, target_season, predictors_input)
    lat=axis_global['lat'];lon=axis_global['lon']
    area_unigrid = compute_unigrid_area(lat)

    # compute GW index
    normvar_2D_model_hist=predvars_2D_model_hist['GWI']; normvar_2D_model_ssp=predvars_2D_model_ssp['GWI']

    region_normvar=predictors_input['region'][predictors_input['predictor'].index('GWI')]

    normvar_2D_model_hist=remove_outliers_dic(normvar_2D_model_hist,area_unigrid,axis_global,'NH_midpolar')
    normvar_2D_model_ssp=remove_outliers_dic(normvar_2D_model_ssp,area_unigrid,axis_global,'NH_midpolar')

    normvar_2D_model_deltaclim=operator_for_perturbation(subtract_function,normvar_2D_model_hist,normvar_2D_model_ssp)    
    normvar_region_model_deltaclim=operator_for_members(spatial_average,normvar_2D_model_deltaclim,area_unigrid,lon,lat,region_normvar,'avg')

    normvar_region_model_deltaclim=operator_for_members(convert_num_to_array,normvar_region_model_deltaclim,0)

    # compute anomalies
    targetvar_2D_model_hist=remove_outliers_dic(targetvar_2D_model_hist,area_unigrid,axis_global,'NH_midpolar')
    targetvar_2D_model_ssp=remove_outliers_dic(targetvar_2D_model_ssp,area_unigrid,axis_global,'NH_midpolar')
    targetvar_2D_model_deltaclim=operator_for_perturbation(subtract_function,targetvar_2D_model_hist,targetvar_2D_model_ssp)
 
    # select target region
    [lat_region, lon_region, bound_box] = regional_outline(lat,lon,target_region)
    targetvar_2D_model_deltaclim_region = operator_for_members(regional_cropping,targetvar_2D_model_deltaclim,bound_box)
    axis_region={'lon':lon_region, 'lat':lat_region}

    # Normalize by Global Warming index
    if norm_GW==True:
        targetvar_2D_model_deltaclim_region = normalize_var(targetvar_2D_model_deltaclim_region,normvar_region_model_deltaclim)

    # compute predictors
    predvars_region_model_deltaclim = {}
    for i in np.arange(0,len(predictors_input['predictor']),1):
        predvar_2D_model_hist=predvars_2D_model_hist[predictors_input['predictor'][i]]
        predvar_2D_model_ssp=predvars_2D_model_ssp[predictors_input['predictor'][i]]
        
        # compute spatial average
        region_=predictors_input['region'][i]
        predvar_2D_model_hist=remove_outliers_dic(predvar_2D_model_hist,area_unigrid,axis_global,'NH_midpolar')
        predvar_2D_model_ssp=remove_outliers_dic(predvar_2D_model_ssp,area_unigrid,axis_global,'NH_midpolar')

        if predictors_input['predictor'][i]=='SIE_Arctic':
            predvar_region_model_hist=operator_for_members(spatial_average,predvar_2D_model_hist,area_unigrid,lon,lat,region_,'int')
            predvar_region_model_ssp=operator_for_members(spatial_average,predvar_2D_model_ssp,area_unigrid,lon,lat,region_,'int')

        else:            
            predvar_region_model_hist=operator_for_members(spatial_average,predvar_2D_model_hist,area_unigrid,lon,lat,region_,'avg')
            predvar_region_model_ssp=operator_for_members(spatial_average,predvar_2D_model_ssp,area_unigrid,lon,lat,region_,'avg')
        predvar_region_model_deltaclim=operator_for_perturbation(subtract_function,predvar_region_model_hist,predvar_region_model_ssp)

        # Normalize by Global Warming index
        if (norm_GW==True) and (predictors_input['predictor'][i]!='GWI'):
            predvar_region_model_deltaclim = normalize_var(predvar_region_model_deltaclim,normvar_region_model_deltaclim)

        # remove Multi-Model Mean
        if (rm_MMM==True) and (predictors_input['predictor'][i]!='GWI'):
            predvar_region_model_deltaclim = rm_MMM_var(predvar_region_model_deltaclim)

        # normalize by Multi-Model STD
        if (scaled_MM==True) and (predictors_input['predictor'][i]!='GWI'):
            predvar_region_model_deltaclim = dv_STD_var(predvar_region_model_deltaclim)

        predvars_region_model_deltaclim[predictors_input['predictor'][i]]=predvar_region_model_deltaclim

        [targetvar_2D_common_model_deltaclim_region, predvars_region_common_model_deltaclim] = find_common_models(targetvar_2D_model_deltaclim_region, predvars_region_model_deltaclim)

    return targetvar_2D_common_model_deltaclim_region, predvars_region_common_model_deltaclim, axis_region

def prepare_ML_init(expt, target_varname, target_season, target_region, predictors_input=predictors_input_list):

    # load target variable
    [targetvar_2D_model, axis, targetvar_models_list] = read_cmip6_clim_model_season(expt, target_varname, target_season) 

    # load predictor variables [contain normalization variable]
    predvars_2D_model={};
    for i in np.arange(0,len(predictors_input['predictor']),1):
        print(predictors_input['predictor'][i])
        [predvar_2D_model, _, predvar_models_list] = read_cmip6_clim_model_season(expt, predictors_input['varname'][i], predictors_input['season'][i])
        predvars_2D_model.update({predictors_input['predictor'][i]:predvar_2D_model})

    return targetvar_2D_model, predvars_2D_model, axis

def compute_pred_corrtable(expt_ctrl, expt_pert, predictors_input=predictors_input_list):

    # NOT scaled by global warming index
    [targetvar_2D_common_model_deltaclim, predvars_region_common_model_deltaclim, axis_region] = prepare_ML_input(expt_ctrl, expt_pert, dummy_varname, dummy_season, dummy_region, predictors_input, False, False)

    common_models_list=list(targetvar_2D_common_model_deltaclim.keys())
    predvars_region_ensmean_deltaclim_array = convert_pred_dic_to_ensmean_array(predvars_region_common_model_deltaclim,common_models_list)

    predvars_nans=np.mean(predvars_region_ensmean_deltaclim_array,axis=0)
    predvars_nans_index=np.where(~np.isnan(predvars_nans))
    predvars_region_ensmean_deltaclim_array=np.squeeze(predvars_region_ensmean_deltaclim_array[:,predvars_nans_index])

    # print Pearson correlation coefficient
    print('NOT SCALED: correlation BEFORE scaling')
    corr_unscaled = np.round(np.corrcoef(predvars_region_ensmean_deltaclim_array),2);
    
    # Scaled by Global warming index
    [targetvar_2D_common_model_deltaclim, predvars_region_common_model_deltaclim, axis_region] = prepare_ML_input(expt_ctrl, expt_pert, dummy_varname, dummy_season, dummy_region, predictors_input, True, True)

    targetvar_models_list=list(targetvar_2D_common_model_deltaclim.keys())    
    predvars_region_ensmean_deltaclim_array=convert_pred_dic_to_ensmean_array(predvars_region_common_model_deltaclim,targetvar_models_list)
    predvars_region_ensmean_deltaclim_array=np.squeeze(predvars_region_ensmean_deltaclim_array)

    predvars_nans=np.mean(predvars_region_ensmean_deltaclim_array[1:,:],axis=0)
    predvars_nans_index=np.where(~np.isnan(predvars_nans))
    predvars_region_ensmean_deltaclim_array=np.squeeze(predvars_region_ensmean_deltaclim_array[:,predvars_nans_index])

    # print Pearson correlation coefficient
    print('SCALED: correlation AFTER scaling')
    corr_scaled = np.round(np.corrcoef(predvars_region_ensmean_deltaclim_array),2); 

    return corr_scaled, corr_unscaled

def compute_eof_storylines(expt_ctrl, expt_pert, tarname, season, region):

    phases=['-','+']
    prednames=['EOF1','EOF2']
    num_predictor = len(prednames)
    xi = stats.chi2.ppf(0.8,num_predictor)    
    t_s = (0.5*xi)**0.5

    # MMM 
    [MMM_change, lon_region, lat_region]=return_clim_ensmean_model(expt_pert, 'MMM', tarname, season, region, expt_ctrl)
    [ML_coeff1, lon_region, lat_region]=return_clim_ensmean_model(expt_pert, 'EOF1', tarname, season, region, expt_ctrl)   
    [ML_coeff2, lon_region, lat_region]=return_clim_ensmean_model(expt_pert, 'EOF2', tarname, season, region, expt_ctrl)
    axis_region={'lon':lon_region, 'lat':lat_region}

    Storyline={'name':[],'data':[]}
    for phase1 in phases:
        if phase1=='0':
            s1=0
        else:
            s1=np.sign(int(phase1+'1'))
        for phase2 in phases:
            if phase2=='0':
                s2=0
            else:
                s2=np.sign(int(phase2+'1'))
                        
            Storyline_=MMM_change+s1*ML_coeff1*t_s+s2*ML_coeff2*t_s 
            Storyline_=s1*ML_coeff1*t_s+s2*ML_coeff2*t_s 
            Storyline['name'].append(prednames[0]+phase1+'|'+prednames[1]+phase2)
            Storyline['data'].append(Storyline_)

    return Storyline, axis_region

#######

# models with wrong SI
# models_exclude=['FGOALS-f3-L','GISS-E2-2-G']

def return_clim_ensmean_model(expt, model, varname, season, region, expt_ctrl='dummy', norm=True, norm_varname='tas', norm_season='annual', norm_region='Global'):

    if expt_ctrl!='dummy':
            
        if norm==True:
            [var_2D_model_dic, axis, models_list] = read_cmip6_normdeltaclim_model(expt_ctrl, expt, varname, season, norm_varname, norm_season, norm_region)
        else:
            [var_2D_model_dic, axis, models_list] = read_cmip6_deltaclim_model(expt_ctrl, expt, varname, season)

        lon=axis['lon']; lat=axis['lat']; area_unigrid = compute_unigrid_area(lat)

        var_2D_model_dic = remove_outliers_dic(var_2D_model_dic,area_unigrid,axis,'NH_midpolar')
        var_2D_model_ensmean_dic = compute_models_ensmean_dic(var_2D_model_dic)

        models_to_removed=['FGOALS-f3-L','FGOALS-g3'] # 'GISS-E2-2-G' 
        for model_name in models_to_removed:
            try:
                del var_2D_model_ensmean_dic[model_name]
            except:
                print('no data for ', model_name)

        [var_2D_model, models_out_list]=convert_dic_to_array_2D(var_2D_model_ensmean_dic)

        if model=='MMM':
            var_2D = np.nanmean(var_2D_model,axis=0)
            [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
            var_2D_region = regional_cropping(var_2D, bound_box)
        elif 'EOF' in model:
            k_eof = int(model[-1])-1
            [EOFs_2D, pcs_1D, varnorm] = return_eof_modes(var_2D_model,axis,region)
            var_2D_region = np.squeeze(EOFs_2D[k_eof,:,:])
            print(model + ' explains ' + str(round(np.diag(varnorm)[k_eof]*100.0)) + '% of variance' )
            [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
        else:
            try:
                ind_model=models_out_list.index(model)
            except:
                print('no data for ' + model)
                return
            var_2D=var_2D_model[ind_model,:]
            [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
            var_2D_region = regional_cropping(var_2D, bound_box)
    else:

        [var_2D_model_dic, axis, models_list] = read_cmip6_clim_model_season(expt, varname, season)
        lon=axis['lon']; lat=axis['lat']; area_unigrid = compute_unigrid_area(lat)

        var_2D_model_dic = remove_outliers_dic(var_2D_model_dic,area_unigrid,axis,'NH_midpolar')
        var_2D_model_ensmean_dic = compute_models_ensmean_dic(var_2D_model_dic)
        [var_2D_model, models_out_list] = convert_dic_to_array_2D(var_2D_model_ensmean_dic)

        if model=='MMM':
            var_2D = np.nanmean(var_2D_model,axis=0)
            [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
            var_2D_region = regional_cropping(var_2D, bound_box)
        elif 'EOF' in model:
            k_eof = int(model[-1])-1
            [EOFs_2D, pcs_1D, varnorm] = return_eof_modes(var_2D_model,axis,region)
            var_2D_region = np.squeeze(EOFs_2D[k_eof,:,:])
            print(model + ' explains ' + str(round(np.diag(varnorm)[k_eof]*100.0)) + '% of variance' )
            [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
        else:
            try:
                ind_model=models_out_list.index(model)
            except:
                print('no data for ' + model)
                return
            var_2D=var_2D_model[ind_model,:]
            [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
            var_2D_region = regional_cropping(var_2D, bound_box)


    return var_2D_region, lon_region, lat_region

def read_cmip6_normdeltaclim_model(expt_ctrl, expt_pert, varname, season, norm_varname, norm_season, norm_region):

    [var_2D_model_deltaclim,axis,_] = read_cmip6_deltaclim_model(expt_ctrl, expt_pert, varname, season)
    lat=axis['lat']; lon=axis['lon']; area_unigrid = compute_unigrid_area(lat)
    [normvar_2D_model_deltaclim,_,_] = read_cmip6_deltaclim_model(expt_ctrl, expt_pert, norm_varname, norm_season)

    # find index of existing models for each list
    var_models_list=list(var_2D_model_deltaclim.keys())
    normvar_models_list=list(normvar_2D_model_deltaclim.keys())
    common_model_list_s=list(set(var_models_list).intersection(normvar_models_list))
    common_model_list=[]
   # re-order
    for model in var_models_list:
        if model in common_model_list_s:
            common_model_list.append(model)
 
    var_2D_common_model_deltaclim = extract_models_dic(var_2D_model_deltaclim,common_model_list)
    normvar_2D_common_model_deltaclim_ = extract_models_dic(normvar_2D_model_deltaclim,common_model_list)
    normvar_region_common_model_deltaclim = operator_for_members(spatial_average,normvar_2D_common_model_deltaclim_,area_unigrid,lon,lat,norm_region,'avg')

    var_2D_common_model_normdeltaclim={}
    for model in common_model_list:
        var_member_list=list(var_2D_common_model_deltaclim[model].keys())
        norm_member_list=list(normvar_region_common_model_deltaclim[model].keys())
        member_list=list(set(var_member_list).intersection(norm_member_list))        
        var_2D_common_model_normdeltaclim[model]={}
        for member in member_list:
            var_2D_common_model_normdeltaclim[model][member]=var_2D_common_model_deltaclim[model][member]/normvar_region_common_model_deltaclim[model][member]

    return var_2D_common_model_normdeltaclim, axis, common_model_list

def read_cmip6_deltaclim_model(expt_ctrl, expt_pert, var, month):

    [var_2D_model_season_ctrl, axis_unigrid, model_list_ctrl] = read_cmip6_clim_model_season(expt_ctrl, var, month)
    [var_2D_model_season_pert, axis_unigrid, model_list_pert] = read_cmip6_clim_model_season(expt_pert, var, month)

    delta_var_2D_common_model_season=operator_for_perturbation(subtract_function,var_2D_model_season_ctrl,var_2D_model_season_pert)
    common_model_list_s=list(set(model_list_ctrl).intersection(model_list_pert))
    # re-order model 
    common_model_list=[]
    for i in range(len(model_list_pert)):
        if model_list_pert[i] in common_model_list_s:
            common_model_list.append(model_list_pert[i])

    return delta_var_2D_common_model_season,axis_unigrid,common_model_list

def return_ensmean_from_members(var_members_model):

    models_list=list(var_members_model.keys())
    output_model={}
    for model in models_list:
        var_members_model_=var_members_model[model]
        members=list(var_members_model_.keys())
        output_array=[]
        for member in members:
            var_member=np.asarray(var_members_model[model][member])
#            if var_member==inf:
#                var_member=np.nan
            output_array.append(var_member)
        output_model[model]=np.nanmean(output_array)

    return output_model

def create_landocean(mask_flag,region='Global'):

    [var_model_season_unigrid, axis_unigrid, models_list] = read_cmip6_clim_model('historical', 'tos', 'summer')

    if mask_flag=='Ocean':
#        model='GISS-E2-1-H'
#        member='r1i1p1f2'
        model='MPI-ESM1-2-LR'
        member='r10i1p1f1'
    elif mask_flag=='Land':
        model='MPI-ESM1-2-LR'
        member='r10i1p1f1'

    var_2D = var_model_season_unigrid[model][member]
    
    mask_2D=np.ones(np.shape(var_2D))
    mask_2D[np.where(np.isnan(var_2D)==True)]=np.nan

    if mask_flag=='Land':
        mask_2D=np.subtract(np.ones(np.shape(mask_2D)),mask_2D)
        
    if region!='Global':
        lat=axis_unigrid['lat'];lon=axis_unigrid['lon']
        [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
        mask_2D = regional_cropping(mask_2D, bound_box)        

    return mask_2D
