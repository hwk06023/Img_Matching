# One shot learning

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

<br/><br/><br/><br/>

