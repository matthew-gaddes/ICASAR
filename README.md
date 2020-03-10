# ICASAR
An algorithm for robustly appling sICA to InSAR data in order to isolate deformation signals and atmospheric signals.  Its use is detailed in the [GitHub Wiki](https://github.com/matthew-gaddes/ICASAR/wiki).  

[Gaddes et al., 2019](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB017519) details its application to Sierra Negra (Galapagos Archipelago, Ecuador), where it was able to isolate the caldera floor uplift signal (source 1), and a topograhically correlated atmospheric phase screen (source 2):


![figure_5](https://user-images.githubusercontent.com/10498635/75799672-4c892c80-5d70-11ea-80a1-749aac2b89d2.png)

<br><br/>

The algorithm also provides tools to visualise the recovered sources in order for a user to determine how robust they are.  Note that the sources capturing the caldera floor uplift (blue points), form an isolated and compact cluster:

![figure_4](https://user-images.githubusercontent.com/10498635/75799539-206dab80-5d70-11ea-9ebe-5ebdd5cf94af.png)

