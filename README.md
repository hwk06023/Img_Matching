# Img_Recognition

## Abstract

Recognize the image from the image to determine if it exists. <br/>
It presents a variety of methodologies for this <br/>
The final goal is process on real-time & Incremental the image to recognize <br/>

I record various attempts in this repo <br/><br/>

## Feature Detection & Matching - opencv

Use SIFT, SURF, ORB, FAST, BRISK, AKAZE .. <br/>
[Comparative analysis](https://ieeexplore.ieee.org/document/8346440)

If features are simple, use FAST, BRISK .. <br/>
else(complex), use SIFT, SURF, AKAZE .. <br/>

Check [Feature_DetectMatch.py](Feature_DetectMatch.py)

![fea_detmat](readme/Fea_DetectMat.png)

In Smart_Camera(Navigation) project,

![fe2](readme/navi_feat1.png)

![fe3](readme/navi_feat2.png)

This project is demanded working robustly(whether a small image is rotated, moved, or changed in brightness) <br/>
So, I'm doing middle processing to boolean the result from feature detection & matching. <br/>

### Homography

### BFmatching

## Matching Template - opencv

Use opencv's matchTemplate, get loc, normalize .. <br/>

Check [template_matching.py](https://github.com/hwk06023/Img_Recognition/blob/main/template_matching.py)
<br/>

![Match template](readme/matchTemp.png)

![template navi](readme/navi_template.png)


In this case, performance is very nice. <br/>
But, size or rotational transformations do not work well, and slow. <br/>
So, I can't use it <br/>


<br/>


## One shot learning

An attempt to overcome the vulnerability of the real world <br/>

- [Siamese Neural Networks](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [CLIPSeg](https://arxiv.org/pdf/2103.00020.pdf)

###  Siamese Neural Networks (Conv)

I use [Huggingface](https://huggingface.co/keras-io/siamese-contrastive), [keras.io](https://keras.io/examples/vision/siamese_contrastive/). <br/>

[Siamese_net](siamese_net.ipynb) <br/>

um.. I miss. useless <br/>


### CLIPSeg

I use [Huggingface](https://huggingface.co/blog/clipseg-zero-shot). <br/>

![clipseg](readme/clipseg.png) <br/>

Check [CLIPSeg.ipynb](https://github.com/hwk06023/Img_Recognition/blob/main/CLIPSeg.ipynb) <br/>

zero-shot learning test's result is good. <br/>

![zero-shot](readme/zero-shot.png) <br/>

But, On the one-shot learning, The two pictures are about me with different backgrounds, I want Recognize me, but this processer recognize my clothes .. <br/>


![zero-shot](readme/one-shot.png) <br/>

Um .. I think because my skin color is similar to the background color. <br/>

<br/>

## Augmetation + Few shot learning

An attempt to improve performance in one-shot learning <br/>
I make use of [paperwithcode's git](https://paperswithcode.com/paper/prototypical-networks-for-few-shot-learning)

#### An attempt (Update)
- Metric based learning - [Prototypical Network](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf), Relation Network, .. <br/>
- Model based learning - .. <br/>
- Optimizer learning - .. <br/>
<br/>

### Augemtation

Flipping, Gray scale, Brightness, Rotation ..

Based [this Repo](https://github.com/hwk06023/Augmentation)


### Prototypical Network




### Relation Network



## + Continual learning

Continual learning is required because the task to be processed is constantly updated. <br/>

Based [this Repo](https://github.com/hwk06023/Continual-Learning) <br/>

So, { n-way, n+1-way, n+2-way, ... }