# Img_Recognition

## Abstract

Recognize the image from the image to determine if it exists. <br/>
It presents a variety of methodologies for this <br/>
The final goal is process on real-time & Incremental the image to recognize <br/><br/>

## Matching Template - opencv

Use opencv's 'cv2.matchTemplate', get loc, normalize .. <br/>

Check [template_matching.py](https://github.com/hwk06023/Img_Recognition/blob/main/template_matching.py)
<br/>

Size, rotational transformations do not work well, and slow


<br/>

## One shot learning (CLIPSeg ..)

An attempt to overcome the vulnerability of the real world <br/>
I use [Huggingface](https://huggingface.co/blog/clipseg-zero-shot). <br/>

![clipseg](readme/clipseg.png) <br/>

Check [CLIPSeg.ipynb]() <br/>

Um.. soso. <br/>

<br/>

## Augmetation+ Few shot learning

An attempt to improve performance in one-shot learning <br/>

#### An attempt (Update)
Metric based learning - Prototypical Network, Relation Network, .. <br/>
Model based learning - .. <br/>
Optimizer learning - .. <br/>

<br/>

### Augemtation


### Prototypical Network



### Relation Network



## + Continual learning

Contilnual learning is required because the task to be processed is constantly updated. <br/>

So, {n-way, n+1-way, n+2-way, ...} lol <br/>