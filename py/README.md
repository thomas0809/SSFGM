# SSFGM (python)

Code for SSFGM with the TCS algorithm. Based on the implementation of https://github.com/matenure/FastGCN.

## Requirement

Python 3.6
Tensorflow 1.4

## Data

Download the input files from 

* [Twitter (World)](http://rosetta6.csail.mit.edu/location_inference_data/twitter_world.zip) (2.2G)
* [Twitter (USA)](http://rosetta6.csail.mit.edu/location_inference_data/twitter_usa.zip) (139M)
* [Weibo](http://rosetta6.csail.mit.edu/location_inference_data/weibo.zip) (616M)

Put them in the `./data` folder.

## Run

For running, the command is 

```
python gcncrf_transductive.py [options]
    --model  crf/gcn           (crf stands for SSFGM with TCS)
    --learning_rate  {value}
    --dataset  twitter_usa/twitter_world/weibo
    --output  {filename}
    --deep                     (enable the deep factor in crf)
```

See the codes for details.

## Example

```
python gcncrf_transductive.py --model crf --learning_rate 0.01 --dataset twitter_usa --output pred_TCS_state.txt
python gcncrf_transductive.py --model gcn --learning_rate 0.01 --dataset twitter_usa --output pred_GCN_state.txt
```
