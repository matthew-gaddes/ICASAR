# ICASAR
An algorithm for robustly appling sICA to InSAR data in order to isolate deformation signals and atmospheric signals.  Its use is detailed in the [GitHub Wiki](https://github.com/matthew-gaddes/ICASAR/wiki).  

[Gaddes et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB017519) details its application to Sierra Negra (Galapagos Archipelago, Ecuador), where it was able to isolate the caldera floor uplift signal (source 1), and a topograhically correlated atmospheric phase screen (source 2):


![figure_5](https://user-images.githubusercontent.com/10498635/75799672-4c892c80-5d70-11ea-80a1-749aac2b89d2.png)

<br><br/>

The algorithm also provides tools to visualise the recovered sources in order for a user to determine how robust they are.  Note that the sources capturing the caldera floor uplift (blue points), form an isolated and compact cluster:

![figure_4](https://user-images.githubusercontent.com/10498635/75799539-206dab80-5d70-11ea-9ebe-5ebdd5cf94af.png)

# June 2020 addition:

If longitude and latitude information for the interferograms is available, the independent components that are found can be geocoded and displayed in Google Earth:
![ge_print](https://user-images.githubusercontent.com/10498635/84274640-02aaa200-ab28-11ea-80e1-ed5e21f26528.jpg)


# August 2020 addition:

If both bootsrapped and non-bootstrapped runs are done in the same step, these can be displayed in the same plot.  Note that the non-boostrapped runs generally create very small/tight clusters (as the sources recovered are generally very similar), and it can be difficult to achieve reasonable clutering results due to two scales of clusters present.  In the example below, many sources recovered during bootstrapped runs ('o's) are marked as noise as they lie outside the compact clusters formed by the bootstrapped data ('x's):

![03_clustering_and_manifold_with_bootstrapping](https://user-images.githubusercontent.com/10498635/91584350-6afc6900-e94a-11ea-856b-59f78f799814.png)
