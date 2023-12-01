import numpy as np
from sklearn.decomposition import TruncatedSVD


from Set_Grid import compute_unigrid_area, regional_outline, regional_cropping

####################################### PCA
def return_mca_modes(var1_2D,var2_2D,axis1,axis2,region1,region2):

    #https://atmos.uw.edu/~breth/classes/AS552/matlab/lect/html/MCA_PSSTA_USTA.html
    lat1 = axis1['lat']; lat2 = axis2['lat']
    area1_2D = compute_unigrid_area(lat1); area2_2D = compute_unigrid_area(lat2)
    var1_2D_MMM = np.nanmean(var1_2D,axis=0,keepdims=True)
    var1_2D = var1_2D - var1_2D_MMM
    var2_2D_MMM = np.nanmean(var2_2D,axis=0,keepdims=True)
    var2_2D = var2_2D - var2_2D_MMM

    [var1_area_1D_region_model_, var1_1D_region_model_, area1_1D_region_model_, model_dim1, lat_dim1, lon_dim1] = initialize_eof_analysis(var1_2D, area1_2D, axis1, region1)
    [var2_area_1D_region_model_, var2_1D_region_model_, area2_1D_region_model_, model_dim2, lat_dim2, lon_dim2] = initialize_eof_analysis(var2_2D, area2_2D, axis2, region2)

    common_ind_Val1 = compute_common_grid_nans(var1_1D_region_model_,model_dim1)
    common_ind_Val2 = compute_common_grid_nans(var2_1D_region_model_,model_dim2)

    var1_area_1D_region_model = var1_area_1D_region_model_[:,common_ind_Val1]
    var1_1D_region_model = var1_1D_region_model_[:,common_ind_Val1]
    area1_1D_region_model = area1_1D_region_model_[:,common_ind_Val1]
    area1_1D_region = area1_1D_region_model[0,:]

    var2_area_1D_region_model = var2_area_1D_region_model_[:,common_ind_Val2]
    var2_1D_region_model = var2_1D_region_model_[:,common_ind_Val2]
    area2_1D_region_model = area2_1D_region_model_[:,common_ind_Val2]
    area2_1D_region = area2_1D_region_model[0,:]

    var1_area_1D_region_model_T = np.transpose(var1_area_1D_region_model)
    var1_1D_region_model_T = np.transpose(var1_1D_region_model)

    var1_var2_area_1D_region_model_covar_1D = np.divide(np.dot(var1_area_1D_region_model_T,var2_area_1D_region_model),np.shape(var2_area_1D_region_model)[0]-1)

    [u, Sigma, v] = compute_svd(var1_var2_area_1D_region_model_covar_1D)    

    # compute squared covariance fraction
    s = np.diag(Sigma)
    varnorm = np.power(s,2)/np.sum(np.power(s,2))

    feature_dim1 = np.shape(v)[1]
    feature_dim2 = np.shape(u)[1]

    MCA1_series=np.empty((feature_dim1,model_dim1)); MCA2_series=np.empty((feature_dim2,model_dim2))
    MCA1_pattern=np.empty((feature_dim1,len(common_ind_Val1))); MCA2_pattern=np.empty((feature_dim2,len(common_ind_Val2)))

    # mode x model
    MCA1_proj=np.dot(v.T,var1_area_1D_region_model.T); MCA2_proj=np.dot(u.T,var2_area_1D_region_model.T)
    std1=np.nanstd(MCA1_proj,axis=1); std2=np.nanstd(MCA2_proj,axis=1)

    for n in range(np.shape(MCA1_proj)[0]-1):
        MCA1_series[n,:]=np.divide(MCA1_proj[n,:],std1[n])
        MCA2_series[n,:]=np.divide(MCA2_proj[n,:],std2[n])
        MCA1_pattern[n,:]=np.multiply(v[:,n],std1[n])
        MCA2_pattern[n,:]=np.multiply(u[:,n],std2[n])
        MCA1_pattern[n,:]=np.divide(MCA1_pattern[n,:],area1_1D_region_model[n,:])
        MCA2_pattern[n,:]=np.divide(MCA2_pattern[n,:],area2_1D_region_model[n,:])

    MCA1 = np.empty((feature_dim1,lat_dim1*lon_dim1)); MCA1[:] = np.NaN
    MCA1[:,common_ind_Val1] = MCA1_pattern
    MCA1_2D = MCA1.reshape((feature_dim1,lat_dim1,lon_dim1))

    MCA2 = np.empty((feature_dim2,lat_dim2*lon_dim2)); MCA2[:] = np.NaN
    MCA2[:,common_ind_Val2] = MCA2_pattern
    MCA2_2D = MCA2.reshape((feature_dim2,lat_dim2,lon_dim2))
    
    return MCA1_2D, MCA2_2D, varnorm

####################################### PCA

def return_eof_modes(var_2D,axis,region):

    lat = axis['lat']
    area_2D = compute_unigrid_area(lat)
    var_2D_MMM = np.nanmean(var_2D,axis=0,keepdims=True)
#    var_2D_MMM = np.mean(var_2D,axis=0,keepdims=True)
    var_2D = var_2D - var_2D_MMM

    [var_area_1D_region_model_, var_1D_region_model_, area_1D_region_model_, model_dim, lat_dim, lon_dim] = initialize_eof_analysis(var_2D, area_2D, axis, region)

    common_ind_Val = compute_common_grid_nans(var_1D_region_model_,model_dim)
    var_area_1D_region_model = var_area_1D_region_model_[:,common_ind_Val]
    var_1D_region_model = var_1D_region_model_[:,common_ind_Val]
    area_1D_region_model = area_1D_region_model_[:,common_ind_Val]
    area_1D_region = area_1D_region_model[0,:]
    [u, Sigma, v] = compute_svd(var_area_1D_region_model)
    [var_pcs_1D, std_var_pcs, varnorm] = compute_pcs(u, Sigma, v)
    EOFs_2D = compute_eof(u,std_var_pcs,area_1D_region,lat_dim,lon_dim,common_ind_Val)    
    
    pcs_1D=[]
    for n in range(len(std_var_pcs)):
        pcs_1D.append(np.divide(var_pcs_1D[n,:], std_var_pcs[n]))
    pcs_1D=np.asarray(pcs_1D)
        
    return EOFs_2D, pcs_1D, varnorm

def initialize_eof_analysis(var_2D_model,area_2D,axis,region):

    lon=axis['lon']; lat=axis['lat']
    [lat_region, lon_region, bound_box] = regional_outline(lat,lon,region)
    var_2D_region_model = regional_cropping(var_2D_model,bound_box)

    model_dim = np.shape(var_2D_region_model)[0]
    lat_dim = np.shape(var_2D_region_model)[1]
    lon_dim = np.shape(var_2D_region_model)[2]

    area_2D_model = np.tile(area_2D, (model_dim,1,1))
    area_2D_region_model = regional_cropping(area_2D_model,bound_box)

    var_area_2D_region_model = np.multiply(var_2D_region_model,area_2D_region_model)

    # format input for PCA
    var_area_1D_region_model = var_area_2D_region_model.reshape((model_dim,lat_dim*lon_dim))
    var_1D_region_model = var_2D_region_model.reshape((model_dim,lat_dim*lon_dim))
    area_1D_region_model = area_2D_region_model.reshape((model_dim,lat_dim*lon_dim))

    return var_area_1D_region_model, var_1D_region_model, area_1D_region_model, model_dim, lat_dim, lon_dim

def compute_common_grid_nans(var_1D_region_model,model_dim):

    # set NaNs to gridcell if one model has NaN
    common_ind_Val=[]
    for i in np.arange(0,model_dim,1):
        common_ind_Val_=np.squeeze(np.argwhere(np.isnan(var_1D_region_model[i,:])==False))
        if not common_ind_Val:
            common_ind_Val=common_ind_Val_
        common_ind_Val=list(set(common_ind_Val).intersection(common_ind_Val_))

    return common_ind_Val

def compute_svd(var_area_1D_region_model):

#    [u, v, s] = return_EOFs(var_area_1D_region_model)
    PC_comp=10

    var_2D = np.transpose(var_area_1D_region_model)
    nComp = min(min(var_2D.shape)-1,PC_comp)
    var_pca = TruncatedSVD(n_components=nComp)
    var_pca.fit(var_2D)
    v_T = var_pca.components_
    v = np.transpose(v_T)
    u = var_pca.transform(var_2D)
    Sigma = var_pca.singular_values_
    inv_s = np.linalg.inv(np.diag(Sigma)) 
    u = np.dot(u,inv_s)

    return u, Sigma, v

def compute_pcs(u, Sigma, v):

    # compute squared covariance fraction
    s = np.diag(Sigma)
    varnorm = np.power(s,2)/np.sum(np.power(s,2))

    # Compute PCs
    var_pcs_1D = np.dot(s,v.T)
    std_var_pcs = np.transpose(np.std(var_pcs_1D,axis=1))

    return var_pcs_1D, std_var_pcs, varnorm

def compute_eof(u,std_var_pcs,area_1D_region,lat_dim,lon_dim,common_ind_Val):

    # Compute EOFs
    var_modes_eof = [];
#    for t in np.arange(0,np.shape(std_var_pcs)[0],1):        
    for t in np.arange(0,np.shape(u)[1],1):        
        var_eof = np.multiply(np.squeeze(u[:,t]), std_var_pcs[t])
        var_modes_eof.append(var_eof)
    var_modes_eof = np.asarray(var_modes_eof);

    # Normalize by surface area
    for n in range(np.shape(var_modes_eof)[0]):
        var_modes_eof[n,:]=np.divide(var_modes_eof[n,:],area_1D_region)  

    feature_dim = np.shape(var_modes_eof)[0]
    EOFs = np.empty((feature_dim,lat_dim*lon_dim)); EOFs[:] = np.NaN
    EOFs[:,common_ind_Val] = var_modes_eof
    EOFs_2D = EOFs.reshape((feature_dim,lat_dim,lon_dim))

    return EOFs_2D
