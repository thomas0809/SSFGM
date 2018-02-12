# SSFGM (C++)


This is a C++ implementation of learning Semi-Supervised Factor Graph Model, including Loopy Belief Propagation, SampleRank, and TCS (with Metropolis-Hastings sampling). If you have any problem with the code and data format, please contact the author by yujieq@csail.mit.edu.

## Requirements
g++ 5.4.0 (requires openmp, c++-11)

## Compile & Run

For compiling, simply use `make`.

For running, the command is 

```
SSFGM -est [options]
	-method [LBP/MH/TCMH]: specify learning algorithm, where MH stands for SampleRank, TCMH stands for TCS wih MH
	-trainfile [filename]: training file
	-srcmodel [filename]: input initialized model file (optional)
	-state [filename]: input initialized state file (optional)
	-dstmodel [filename]: output model file
	-pred [filename]: output prediction file
	-niter [number]: the maximum iterations in training
	-ninferiter [number]: the maximum iterations in evaluation/infernce
	-lrate [number]: learning rate η
	-eval [number]: δ, evaluate the model after each δ iterations
	-earlystop [number]: ε, stop learning if validation accuracy does not increase for ε evaluations
	-batch [number]: batch size
	-thread [number]: number of threads
```

Example

```
./SSFGM -est -method MH -trainfile input_weibo.txt -lrate 0.1 -niter 1000000 -ninferiter 1000000 -batch 5000 -eval 1000 -thread 4 -earlystop 20 -pred pred_weibo_MH.txt
```

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
where `line_a`, `line_b` correspond to two nodes in the first part, and lines are counted starting with 1. `edgetype` is a string indicating the type of this edge. Currently the code only support one type.

## Datasets

Weibo: https://aminer.org/influencelocality

Facebook: http://snap.stanford.edu/data/egonets-Facebook.html
