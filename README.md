# ICASAR
An algorithm for robustly appling sICA to InSAR data in order to isolate deformation signals and atmospheric signals.  Its use is detailed in the [GitHub Wiki](https://github.com/matthew-gaddes/ICASAR/wiki).  

[Gaddes et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB017519) details its application to Sierra Negra (Galapagos Archipelago, Ecuador), where it was able to isolate the caldera floor uplift signal (source 1), and a topograhically correlated atmospheric phase screen (source 2):


![figure_5](https://user-images.githubusercontent.com/10498635/75799672-4c892c80-5d70-11ea-80a1-749aac2b89d2.png)

<br><br/>

The algorithm also provides tools to visualise the recovered sources in order for a user to determine how robust they are.  Note that the sources capturing the caldera floor uplift (blue points), form an isolated and compact cluster:

![figure_4](https://user-images.githubusercontent.com/10498635/75799539-206dab80-5d70-11ea-9ebe-5ebdd5cf94af.png)

# September 2020 addition:
ICASAR now supports temporal data.  Whilst this requires no changes to the heart of the algorithm, it required many changes to the plotting functionality, and the creation of a temporal example.  The interactive plot (when hovering over a data point, more information is presented in an inset axes) is now a stand-alone function, for use with any data ([link](https://github.com/matthew-gaddes/interactive_2d_plot)).  

Presented with a set of mixtures created from an unknown number (but assumed to be less than the number of mixtures) of unknown sources:
![temporal_mixtures](https://user-images.githubusercontent.com/10498635/93112344-cdc26400-f6af-11ea-95b3-aedbae3bf095.png)

The ICASAR algorithm returns an interactive figure which suggests that there are three latent sources (the compact and isolated clusters), and a noise term that is being interpreted as two clusters:

![interative_clustering](https://user-images.githubusercontent.com/10498635/93112519-05c9a700-f6b0-11ea-815f-5781f369d2af.gif)

The three latent sources and two noise sources can be visualised.  For further analysis, the noise terms can be discarded.  
![temporal_ICASAR](https://user-images.githubusercontent.com/10498635/93112719-3ad5f980-f6b0-11ea-857e-eea2d94cbfe7.png)

This contrasts with the results of FastICA, which doesn't provide any information as to which recovered sources capture the latent sources, and which capture the noise terms.  
![temporal_FastICA](https://user-images.githubusercontent.com/10498635/93112914-7244a600-f6b0-11ea-8447-6ac3a45705fb.png)





# August 2020 addition:

If both bootsrapped and non-bootstrapped runs are done in the same step, these can be displayed in the same plot.  Note that the non-boostrapped runs generally create very small/tight clusters (as the sources recovered are generally very similar), and it can be difficult to achieve reasonable clutering results due to two scales of clusters present.  In the example below, many sources recovered during bootstrapped runs ('o's) are marked as noise as they lie outside the compact clusters formed by the bootstrapped data ('x's):

![03_clustering_and_manifold_with_bootstrapping](https://user-images.githubusercontent.com/10498635/91584350-6afc6900-e94a-11ea-856b-59f78f799814.png)

# June 2020 addition:

If longitude and latitude information for the interferograms is available, the independent components that are found can be geocoded and displayed in Google Earth:
![ge_print](https://user-images.githubusercontent.com/10498635/84274640-02aaa200-ab28-11ea-80e1-ed5e21f26528.jpg)

# April 2021 addition:
To deal with small signals (i.e. ones that are not visible in a single 12 day Sentinel-1 interferogram), we can aid the isolation of these signals through computing all the possible interferograms between the acquisitions (i.e. to include many longer temporal baseline interferograms).  This allows for the correlations between the strength with which an IC is used for a given interferograms, and its temporal baseline.  ICs that contain deformation are likely to be used strongly in long temporal baseline interferograms.  This figure also computes the correlations between ICS and the DEM, which is useful for determining if a signal captures a topographically correlated atmospheric phase screen.  

![ICs_and_correlations](https://user-images.githubusercontent.com/10498635/115881889-2034e180-a444-11eb-8eb2-4e090653413c.png)


