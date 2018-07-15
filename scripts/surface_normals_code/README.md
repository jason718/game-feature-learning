# Generate surface normal from depth maps

This code is borrowed from Github repo: https://github.com/aayushbansal/MarrRevisited.git by Aayush Bansal. Below is his guidance about this code:

We used the code from [Wang et al. (CVPR 2015)](http://www.cs.cmu.edu/~xiaolonw/deep3d.html) for raw video frames as it was computationally too expensive to compute the normals using the approach of [Ladicky et al.](https://www.inf.ethz.ch/personal/ladickyl/normals_eccv14.pdf) on all video frames. The codes can be downloaded using
   ```make
   # Code to extract surface normal maps from kinect using Wang et al. (CVPR 2015)
   # Details to use this code are given in the folder
   cd toolbox/
   wget http://www.cs.cmu.edu/~aayushb/marrRevisited/data/kinect_normals_code.tar.gz
   tar -xvf kinect_normals_code.tar.gz
   cd ..
   # For more details about this code, see the script getNormals.m
   # All the required data is present in this folder
   ```

## SceneNet script I was using
getNormals_scenenet.m
