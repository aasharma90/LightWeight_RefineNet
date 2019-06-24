# Light-Weight RefineNet (in PyTorch) (Unofficial/Modified)

This repository provides unofficial/modified code for LightWeight-RefineNet from the paper `Light-Weight RefineNet for Real-Time Semantic Segmentation`, available [here](http://bmvc2018.org/contents/papers/0494.pdf)

```
Light-Weight RefineNet for Real-Time Semantic Segmentation
Vladimir Nekrasov, Chunhua Shen, Ian Reid
In BMVC 2018
```
Official code can be picked from the author's implementation available [here](https://github.com/DrSleep/light-weight-refinenet)

# Changes w.r.t. the official code
I've modified the code to work on CityScapes dataset, and also visualize the predicted labels. Using the training configuration in ```src/config.py```, I could obtain the best valdiation score of 75.28% on this dataset. For near replication, you can either -

1) Train the model yourself. Simply run ```source train.sh``` 
2) Download the pre-trained checkpoint from [here](https://www.dropbox.com/s/e7jqd3r3frfvd1r/checkpoint.pth.tar?dl=0) and place it at ```ckpt/run_20190219/```. Then, simply run ```source test.sh```. It will produce the aforementioned score and also save the predicted labels at ```outputs/run_20190219/cityscapes/``` 

For either, ensure you've downloaded the dataset first (from [here](https://www.cityscapes-dataset.com/downloads/)) and modified the directory params accordingly (in ```src/config.py``` or ```src/test.py```). Feel free to experiment with other training configurations, and kindly inform me if you can achieve a better validation/test score.

