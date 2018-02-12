# SSFGM

This repository contains the codes for the Semi-Supervise Factor Graph Model (SSFGM).

If you have any problem with the code and data format, please contact the author by yujieq@csail.mit.edu.


## Codes

<<<<<<< HEAD
In the `./SSFGM-py` folder, we present our python implementation of the Two-Chain Sampling (TCS) algorithm, based on Tensorflow framework. (newest version)

In the `./SSFGM-cpp` folder, we present our c++ implementation of the Loopy Belief Propagation (LBP), SampleRank, and TCS with Metropolis-Hastings sampling algorithm. 
=======
In the `./PYTHON` folder, we present our python implementation of the Two-Chain Sampling (TCS) algorithm, based on Tensorflow framework. (newest version)

In the `./CPP` folder, we present our c++ implementation of the Loopy Belief Propagation (LBP), SampleRank, and TCS with Metropolis-Hastings sampling algorithm. 
>>>>>>> eb83b31dae940aca4d482e75d10c309a3de4608b

In the `./misc` folder, there are some other scripts for baseline methods or evaluations.

Please see each folder for details.


## Datasets

Please download the preprocessed feature files using the following links:

* [Twitter (World)](http://rosetta6.csail.mit.edu/location_inference_data/twitter_world.zip) (2.2G)
* [Twitter (USA)](http://rosetta6.csail.mit.edu/location_inference_data/twitter_usa.zip) (139M)
* [Weibo](http://rosetta6.csail.mit.edu/location_inference_data/weibo.zip) (616M)

We cannot release the raw data of the Twitter datasets due to some limitations. Original data for Weibo and Facebook can be found at:

Weibo: https://aminer.org/influencelocality

Facebook: http://snap.stanford.edu/data/egonets-Facebook.html



## Data Format

Training file consists of two parts: node and edge.

The first part is node. Each line represent a node (instance), and the format is defined as follows:

```
[+/*/?]label featname_1:val featname_2:val ... [#id]
```
where `+/*/?` each stands for training/validation/testing data, labels and feature names can be strings (length<32). The value can be real-valued or 0/1. We suggest to normalize the input features to [0,1].

The second part is edge. Each line represent an edge (correlation between two instances). The format is:

```
#edge line_a line_b edgetype
```
where `line_a`, `line_b` correspond to two nodes in the first part, and lines are counted starting with 0. `edgetype` is a string indicating the type of this edge. Currently the code only support one type.

