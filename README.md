# Light-Weight RefineNet (in PyTorch)

This repository provides modified code for LightWeight-RefineNet from the paper `Light-Weight RefineNet for Real-Time Semantic Segmentation`, available [here](http://bmvc2018.org/contents/papers/0494.pdf)

```
Light-Weight RefineNet for Real-Time Semantic Segmentation
Vladimir Nekrasov, Chunhua Shen, Ian Reid
In BMVC 2018
```
Official code can be picked from the author's implementation available [here](https://github.com/DrSleep/light-weight-refinenet)

# Changes w.r.t to the official code
I've slightly modified the code to work on CityScapes dataset, and also visualize the predicted labels. Using the training configuration in ```src/config.py```, I obtained the best valdiation score of 75.32% on this dataset. For near replication, you can either -
1) Train the model yourself. Simply run ```source train.sh```. Snapshot directory for your reference is available [here](https://www.dropbox.com/sh/vncb11ad8xbxfq4/AAAuc72RCDC9mMXfdn1-ez98a?dl=0)
2) Download my pre-trained model from here ```source test.sh``` (for testing)

