# Img_Recognition

## Abstract

Recognize the image from the image to determine if it exists. <br/>
Images that are part of an image may have been rotated, moved, or changed in brightness  <br/>
It presents a variety of methodologies for this <br/>
The final goal is process on real-time & Incremental the image to recognize <br/>

I record various attempts in this repo <br/><br/>

## Matching Template - opencv

Use opencv's matchTemplate, get loc, normalize .. <br/>

Check [template_matching.py](https://github.com/hwk06023/Img_Recognition/blob/main/template_matching.py)
<br/>

![Match template](readme/matchTemp.png)

In Smart_Camera(Navigation) project [ Easy case ], 
![template navi](readme/navi_template.png)


In this case, performance is very nice. <br/>
But, size or rotational transformations (hard cases) do not work well, and slow. <br/>
So, I can't use it <br/>


<br/>

## Feature Detection & Matching - opencv

Use SIFT, SURF, ORB, FAST, BRISK, AKAZE .. <br/>
[Comparative analysis](https://ieeexplore.ieee.org/document/8346440)

If features are simple, use FAST, BRISK .. <br/>
else(complex), use SIFT, SURF, AKAZE .. <br/>

Check [Feature_DetectMatch.py](Feature_DetectMatch.py)

![fea_detmat](readme/Fea_DetectMat.png)

In Smart_Camera(Navigation) project [ Easy case ],

![fe2](readme/navi_feat1.png)

![fe3](readme/navi_feat2.png)

In Smart_Camera(Navigation) project [ Hard case ], 

![f2_1](readme/readme_hard_1.png)

This case's performance is not good yet.. <br/><br/>

This project is demanded working robustly(whether a small image is rotated, moved, or changed in brightness) <br/>
So, I'm doing middle processing to boolean the result from feature detection & matching. <br/>

### Homography

I think if i use this one, my app work robustly. <br/>

![wiki_homogr.png](readme/wiki_homogr.png) <br/>

As far as, I know homography works for planar objects <br/>
So, I use before, detect planar objects in small image <br/> <br/>

Ratio = 0.6, Good matches:122/53093
![ratio_0.5](readme/ratio_0.6.png)

Ratio = 0.5, Good matches:20/53093
![ratio_0.5](readme/ratio_0.5.png)

Check [Homography.py](Homography.py)


#### solvePnP

solvePnP is

#### BFmatching

BFmatching is BruteForce matching. <br/>
I get the boolean result by using BFmatching <br/>

If Length of matching >= threshold is True <br/>
else(Length of matching < threshold), False. <br/>

I used BFmatching even though I could use FLANN because accuracy is more important than speed. <br/>

Check [BFmatching.py](BFmatching.py)

## Program Scenarios (Feature Matching)

Based on the above contents, I would like to write it as a program <br/>

![scenarios](readme/pipeline_hand.png)

Check


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


#### reference
https://en.wikipedia.org/wiki/Homography
https://ieeexplore.ieee.org/document/8346440
https://arxiv.org/pdf/2103.00020.pdf
https://paperswithcode.com/paper/prototypical-networks-for-few-shot-learning
https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf
https://keras.io/examples/vision/siamese_contrastive