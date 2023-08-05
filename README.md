# Img_Recognition

## Abstract

Recognize the image from the image to determine if it exists. <br/>
It presents a variety of methodologies for this <br/>
The final goal is process on real-time & Incremental the image to recognize <br/><br/>

## Matching Template - opencv

Use opencv's matchTemplate, get loc, normalize .. <br/>

Check [template_matching.py](https://github.com/hwk06023/Img_Recognition/blob/main/template_matching.py)
<br/>

![Match template](readme/matchTemp.png)

Size, rotational transformations do not work well, and slow


<br/>

## One shot learning (CLIPSeg ..)

An attempt to overcome the vulnerability of the real world <br/>
I use [Huggingface](https://huggingface.co/blog/clipseg-zero-shot). <br/>

![clipseg](readme/clipseg.png) <br/>

Check [CLIPSeg.ipynb](https://github.com/hwk06023/Img_Recognition/blob/main/CLIPSeg.ipynb) <br/>

zero-shot learning test's result is good. <br/>

![zero-shot](readme/zero-shot.png) <br/>

But, On the one-shot learning, The two pictures are about me with different backgrounds, I want Recognize me, but this processer recognize my clothes .. <br/>


![zero-shot](readme/one-shot.png) <br/>

Um .. I think because my skin color is similar to the background color. <br/>

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