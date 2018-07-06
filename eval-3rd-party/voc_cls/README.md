# VOC 2007 Classification

This code is written by Philipp Krähenbühl, the original github repo is: https://github.com/philkr/voc-classification

Jeff Donahue modified above code to support python2: https://github.com/jeffdonahue/voc-classification

To run their code, you need to get Philipp's caffe future branch https://github.com/philkr/caffe/tree/future
(He use a few custom layers, mainly HDF5 related) and compile it.

**Please read above two repo README to get basic information.**

## This Repo
My job is just to put everything together here. And provide the running script I was using.

### Modifcation I made
I was training my model in a fully-convolutional way on synthetic data. To better overcome the domain difference
and adapt the learned features, I changed the learning rate of fc6-fc8 10 times bigger and fine-tune longer.
Please try these to see whether it can help yours.


