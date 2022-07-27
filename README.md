# Vessel-Reconstruction


Program reconstructs the 3D geometry of a simulated blood vessel from a tracked 2D ultrasound scan. 434 frames from an ultrsound scan of the vessel with corresponding ground truth segmentations, as well as all transforms needed to transform points from the image coordinate system to 3D slicer’s RAS coordinate system, and a trained U-Net were provided and used in this project. To refine predictions, various methods were attemped and included in the code.



Before running the code, ensure  the following libraries are be loaded:
1. os
2. re
3. cv2
4. glob
5. errno
6. random
7. numpy
8. pandas
9. PIL
10. matplotlib
11. tensorflow
12. sklearn
