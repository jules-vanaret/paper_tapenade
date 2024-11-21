import tifffile
import numpy as np
import napari
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.optimize import curve_fit
from skimage.measure import label, regionprops
import scipy.ndimage as ndi
from tqdm import tqdm
from pathlib import Path

def column_mask(image,mask,half_size=60): #not run here
    column_mask = np.zeros_like(image,dtype=float)
    rp = regionprops(label(mask))
    rp = sorted(rp,key=lambda x: x.area,reverse=True)
    center = rp[0].centroid
    column_mask[:,int(center[1])-half_size:int(center[1])+half_size,int(center[2])-half_size:int(center[2])+half_size] = 1
    column_mask[column_mask==0]=np.nan
    return column_mask

def intensity_per_plane(image, mask,num_sample:int=0):
    df= pd.DataFrame(columns=['Sample', 'Depth', 'Intensity'])
    float_image=image.astype(float)
    if np.isnan(mask).any()==False:
        float_image[np.invert(mask)]=np.nan
    for z in range(len(image)):
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            intensity = np.nanmean(float_image[z])
            if np.isnan(intensity)==False:
                with warnings.catch_warnings():
                    warnings.filterwarnings(action='ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.')
                    df=df._append({'Sample': num_sample, 'Depth': z+1, 'Intensity': intensity}, ignore_index=True)

    return df
    
def normalize_df_column_to_max(df):
    """Normalize intensity values by maximum."""
    df["Intensity"] /= df["Intensity"].max()
    return df

def cut_first_slices(image,nb:int=0):
    shape = image.shape
    new_image = np.zeros((shape[0]-nb,shape[1],shape[2]))
    new_image=image[nb:,:,:]
    return new_image

def concatenate_df(df,image,num_sample,mask,normalize:bool=True,return_image:bool=False,plans_to_cut:list=[]):
    image = image*mask
    if normalize==True:
        image = cut_first_slices(image,plans_to_cut[num_sample])
        mask = cut_first_slices(mask,plans_to_cut[num_sample])

    df_intensity = intensity_per_plane(image,mask,num_sample=num_sample)
    if normalize==True:
        df_intensity= normalize_df_column_to_max(df_intensity)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.')
        df = pd.concat([df, df_intensity])
    if return_image==True:
        return df,image
    else:
        return df
     
def find_expon_parameters(df):
    mean = df.groupby('Depth').mean()
    intensities = mean['Intensity'].to_numpy()
    depth = mean.index.to_numpy()
    ind_max = np.argmax(intensities)
    intensities = intensities[ind_max:]
    depth = depth[ind_max:]
    def exponential(x, b):
        return np.exp(-x/b)
    popt, pcov = curve_fit(exponential, depth, intensities)
    expon_fit = exponential(depth, *popt)
    return expon_fit,popt,depth

def compute_d_from_df(list_d,df_image,ind):
    list_samples = df_image['Sample'].unique()

    for num_sample in list_samples:
        df_image_sample = df_image[df_image['Sample']==num_sample]
        expon_fit,popt,depth = find_expon_parameters(df_image_sample)
        if plot_individual_fits_bool==True :
            plt.plot(depth, expon_fit,label=str(wavelengths[ind])+'_'+str(num_sample),color=cm.Greens(0.3+num_sample/len(list_samples)))
        list_d[ind].append(popt[0])
    return list_d


def process_samples(df, subfolder, channel, plans_to_cut,column_bool,normalization_bool):
    """Process all samples for a given dataset."""
    samples = tifffile.imread(str(Path(folder) / f"S3_wavelength/{subfolder}/*.tif"))
    for num, image in tqdm(enumerate(samples), desc=f"Processing {subfolder}"):
        if column_bool==True:
            mask = tifffile.imread(Path(folder)/f'S3_wavelength/{subfolder}/crops/{num+1}.tif') #here we read the column mask previously done but this is where the function column_mask should be called
        else :
            mask = tifffile.imread(Path(folder) / f"S3_wavelength/{subfolder}/masks/{num+1}.tif")
        image_channel = image[:, channel, :, :]
        df = concatenate_df(
            df,
            image=image_channel,
            mask=mask,
            num_sample=num,
            normalize=normalization_bool,
            plans_to_cut=plans_to_cut,
        )
    return df

fig,ax = plt.subplots(1,figsize=(10,7))

folder = rf'Z:\EqpLenne2\SHARED\Whole_Mount_paper'
colors = ['#1965B0','#4EB265','#E8601C','#882E72']

wavelengths=[405,488,555,647]
std_ratios = [0,0,0,0]
normalization_bool = True
column_bool=True #if not, we look at the decay in the full sample, not in a central column
plot_individual_fits_bool=False
plans_to_cut_farred = [57,35,44,25]
plans_to_cut_red = [25,35,25,25,25]
plans_to_cut_red_tdt = [15,25,20,25,15,15]
plans_to_cut_green_1 = [30,15,40,35,60,50,30]
plans_to_cut_green_2=[30,30,30,10,40,20,30]
plans_to_cut_blue1 = [16,16,10,10,16]
plans_to_cut_blue2 = [25,30,20,25,30]


dfs = {}
dfs["blue"] = process_samples(
    pd.DataFrame(columns=["Sample", "Depth", "Intensity"]),
    subfolder=f"hoechst_and_spy555",
    channel=0, #using only the blue channel
    plans_to_cut=plans_to_cut_blue1,
    column_bool=column_bool,
    normalization_bool=normalization_bool,
)
dfs["blue"] = process_samples(
    dfs["blue"],
    subfolder=f"hoechst_and_draq5",
    channel=0, #we only use the blue channel, and add the intensities to the previous dataframe, to gather the different hoechst
    plans_to_cut=plans_to_cut_blue2,
    column_bool=column_bool,
    normalization_bool=normalization_bool,
)

dfs["green"] = process_samples(
    pd.DataFrame(columns=["Sample", "Depth", "Intensity"]),
    subfolder=f"488_spydna",
    channel=1,
    plans_to_cut=plans_to_cut_green_1,
    column_bool=column_bool,
    normalization_bool=normalization_bool,
)

dfs["red"] = process_samples(
    pd.DataFrame(columns=["Sample", "Depth", "Intensity"]),
    subfolder=f"555_spydna",
    channel=2,
    plans_to_cut=plans_to_cut_red,
    column_bool=column_bool,
    normalization_bool=normalization_bool,
)

dfs["red"] = process_samples(
    dfs["red"],
    subfolder=f"555_tdt",
    channel=2,
    plans_to_cut=plans_to_cut_red_tdt,
    column_bool=column_bool,
    normalization_bool=normalization_bool,
)

dfs["farred"] = process_samples(
    pd.DataFrame(columns=["Sample", "Depth", "Intensity"]),
    subfolder=f"647_draq5",
    channel=3,
    plans_to_cut=plans_to_cut_farred,
    column_bool=column_bool,
    normalization_bool=normalization_bool,
)
sns.lineplot(data=dfs["blue"], x="Depth", y="Intensity", linewidth=3, color=colors[0], ci="sd")
sns.lineplot(data=dfs["green"], x="Depth", y="Intensity", linewidth=3, color=colors[1], ci="sd")
sns.lineplot(data=dfs["red"], x="Depth", y="Intensity", linewidth=3, color=colors[2], ci="sd")
sns.lineplot(data=dfs["farred"], x="Depth", y="Intensity", linewidth=3, color=colors[3], ci="sd")

plt.xlabel('Depth z (µm)',fontsize=30)
plt.ylabel('Normalized intensity in \n a central crop (A.U.)',fontsize=30)
plt.xticks([0,100,200,300],fontsize=30)
plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=30)

expon_fit_blue,popt_blue,depth_blue = find_expon_parameters(dfs["blue"])
expon_fit_green,popt_green,depth_green = find_expon_parameters(dfs["green"])
expon_fit_red,popt_red,depth_red = find_expon_parameters(dfs["red"]) 
expon_fit_farred,popt_farred,depth_farred = find_expon_parameters(dfs["farred"])

df_list = [(dfs["blue"], 0), (dfs["green"], 1), (dfs["red"], 2), (dfs["farred"], 3)]
list_d = [[] for _ in range(4)]
for df_image, ind in df_list:
    list_d = compute_d_from_df(list_d, df_image, ind)

plt.plot(depth_blue, expon_fit_blue,color=colors[0],linewidth=2,linestyle='--',label='exponential fit hoechst')
plt.plot(depth_green, expon_fit_green, color=colors[1],linewidth=2,linestyle='--',label='exponential fit green')
plt.plot(depth_red, expon_fit_red, color=colors[2],linewidth=2,linestyle='--',label='exponential fit red')
plt.plot(depth_farred, expon_fit_farred, color=colors[3],linewidth=2,linestyle='--',label='exponential fit farred')
plt.xticks([0,100,200,300],fontsize=30)
plt.show()
fig.savefig(Path(folder)/'S3a_plot.svg')

    
fig, ax = plt.subplots(2, 1, figsize=(6, 10))

for i in range(4):
    ax[0].scatter(wavelengths[i]*np.ones(len(list_d[i])),list_d[i],s=20,color=colors[i])
mean_d = [np.mean(list_d[i]) for i in range(4)]
for i in range(4):
    ratios = [mean_d[0]/m for m in list_d[i]]
    std_ratios[i] = np.std(ratios)

ax[0].scatter(wavelengths,mean_d,label='mean',s=40,color='black')
ax[1].errorbar(wavelengths, [mean_d[0]/m for m in mean_d], yerr=std_ratios, fmt='o',color='black')
ax[0].set_xlabel('Wavelength (nm)',fontsize=30)
ax[0].set_ylabel('d (µm)',fontsize=30)
ax[0].set_xticks(wavelengths)
ax[0].tick_params(axis='both', which='major', labelsize=25)
ax[0].set_yticks([50,75,100,125])
ax[1].set_xlabel('Wavelength (nm)',fontsize=30)
ax[1].set_ylabel('d_hoechst/d',fontsize=30)
ax[1].tick_params(axis='both', which='major', labelsize=25)
ax[1].set_xticks(wavelengths)
ax[1].set_yticks([round(mean_d[0]/m,2) for m in mean_d])
plt.legend()
fig.savefig(Path(folder)/'S3_c_d_plot.svg')
plt.show()

napari.run()