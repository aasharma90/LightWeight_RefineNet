# Light-Weight RefineNet (in PyTorch)

This repository provides modified code for LightWeight-RefineNet from the paper `Light-Weight RefineNet for Real-Time Semantic Segmentation`, available [here](http://bmvc2018.org/contents/papers/0494.pdf)

```
Light-Weight RefineNet for Real-Time Semantic Segmentation
Vladimir Nekrasov, Chunhua Shen, Ian Reid
In BMVC 2018
```
Official code can be picked from the author's implementation available [here](https://github.com/DrSleep/light-weight-refinenet)

# Changes w.r.t to the official code
I've slightly modified the code to work on CityScapes dataset. Using the training configuration in ```src/config.py```, I obtained the best valdiation score of 75.32% on this dataset. For near replication, you can do either -
1) Train the model yourself. Simply run 
```source train.sh```
2) Downloa```source test.sh``` (for testing)

