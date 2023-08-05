# Img_Recognition

## Abstract

Recognize the image from the image to determine if it exists. <br/>
It presents a variety of methodologies for this <br/>
The final goal is process on real-time & Incremental the image to recognize <br/><br/>

## Matching Template - opencv

Use opencv's 'cv2.matchTemplate', get loc, normalize .. <br/>

<br/>



Size, rotational transformations do not work well, and slow


<br/>

## AUGMENTATION + Few shot learning (CLIPSeg ..)

An attempt to overcome the vulnerability of the real world
I use Huggingface.

![clipseg](readme/clipseg.png)




### + Continual learning

Contilnual learning is required because the task to be processed is constantly updated.