# CMFB-and-FSF
# CMFB-based fusing saliency features
This is tensorflow implementation of "Cosine Modulated Filter Bank (CMFB-based) architecture for extracting and fusing saliency features"
<h3> Model</h3>

![fig1](https://github.com/ALIMDYOUSUF/CMFB-and-FSF/assets/91628312/fe6c339c-4980-4db6-8ff2-3d6948371403)

![fig2](https://github.com/ALIMDYOUSUF/CMFB-and-FSF/assets/91628312/1599f6e6-98a2-4943-85ff-64ebcd741f7a)

![fig4](https://github.com/ALIMDYOUSUF/CMFB-and-FSF/assets/91628312/861b4674-8a21-4200-b43e-83aefd6fe1cf)

<h3>Pretrained Model</h3>
https://drive.google.com/drive/folders/0B6l9O8aWij8fUGtVNldUTXA4eHc?resourcekey=0-h1RonqWwyH-Z0jgRLVC4mQ&usp=sharing

<h3>Usage</h3>
1. Download pretrained model and put them under folder "salience_model" ,(need to create folder yourself)<br />
2. run code<br />

If you want to test whole folder images, run this:  

```
python inference.py --rgb_folder=[your folder]
```

sample:  

```
python inference.py --rgb_folder=./test
```

If you want to test only one image,run this:  

```
python inference.py --rgb=[your image]
```

sample:

```
python inference.py --rgb=animal1.jpg
```

<h3>Sample</h3>

![Sample 1](https://github.com/ALIMDYOUSUF/CMFB-and-FSF/assets/91628312/859124a2-5b33-4719-ba4d-a9612ed8d25c)

![Sample 2](https://github.com/ALIMDYOUSUF/CMFB-and-FSF/assets/91628312/35e0dff0-e6fa-425f-9900-813c3d2efe13)

![Sample 3](https://github.com/ALIMDYOUSUF/CMFB-and-FSF/assets/91628312/e38456b1-fe1f-4d89-96e0-69276358a07e)

more detail please read source code.


💖 Some great tools can be found at [resource websites](#resource-websites).

> Please **cite the related paper** if you **use their dataset** 😄
>
> there have some datasets in the issue [https://github.com/lartpang/awesome-segmentation-saliency-dataset/issues/15](https://github.com/lartpang/awesome-segmentation-saliency-dataset/issues/15). I hope it works for you.

- [(#datasets)]
  - [Saliency](#saliency)
   
    - [RGB-D Saliency](#rgb-d-saliency)
      - [NLPR/RGBD1000](#nlprrgbd1000)
      - [NJU400/2000](#nju4002000)
      - [STEREO/SSB](#stereossb)
      - [LFSD](#lfsd)
      - [RGBD135/DES](#rgbd135des)
      

### RGB-D Saliency

Thanks:

* @JXingZhao: [https://github.com/JXingZhao/ContrastPrior](https://github.com/JXingZhao/ContrastPrior)
* @jiwei0921: [https://github.com/jiwei0921/RGBD-SOD-datasets](https://github.com/jiwei0921/RGBD-SOD-datasets)
* More Details can be found at: [http://dpfan.net/d3netbenchmark/](http://dpfan.net/d3netbenchmark/)


#### NLPR/RGBD1000

![1546138815074](./assets/1546138815074.png)

* Paper: [Rgbd salient object detection: a benchmark and algorithms](https://docs.google.com/uc?authuser=0&id=0B1wzzt1_uP1rb250d0t6dVFXWG8&export=download)
* Project: [https://sites.google.com/site/rgbdsaliency/home](https://sites.google.com/site/rgbdsaliency/home)
* Download: [https://sites.google.com/site/rgbdsaliency/dataset](https://sites.google.com/site/rgbdsaliency/dataset)

NLPR is also called RGBD1000 dataset which including 1,000 images. There may exist multiple salient objects in each image. The structured light depth images are obtained by the Microsoft Kinect under different illumination conditions.

#### NJU400/2000

![1546139249376](./assets/1546139249376.png)

* Paper:

  + NJU400: [Depth saliency based on anisotropic center-surround difference](http://mcg.nju.edu.cn/publication/2014/icip14-jur.pdf)
  + NJU2000: [Depth-aware salient object detection using anisotropic center-surround difference](http://mcg.nju.edu.cn/publication/2015/spic15-jur.pdf)
* Project:

  * [MGG](http://mcg.nju.edu.cn/index.html)
  * [http://mcg.nju.edu.cn/publication/2014/icip14-jur/index.html](http://mcg.nju.edu.cn/publication/2014/icip14-jur/index.html)
* Download:

  * Official:
    + [http://mcg.nju.edu.cn/resource.html](http://mcg.nju.edu.cn/resource.html)
    + [http://mcg.nju.edu.cn/dataset/nju400.zip](http://mcg.nju.edu.cn/dataset/nju400.zip)
    + [http://mcg.nju.edu.cn/dataset/nju2000.zip](http://mcg.nju.edu.cn/dataset/nju2000.zip)

  + See [http://dpfan.net/d3netbenchmark/](http://dpfan.net/d3netbenchmark/)

NJU2000 contains 2003 stereo image pairs with diverse objects and complex, challenging scenarios, along with ground-truth map. The stereo images are gathered from 3D movies, the Internet, and photographs taken by a Fuji W3 stereo camera.

#### STEREO/SSB

![](assets/2019-05-13-19-48-20.png)

* Paper: [Leveraging stereopsis for saliency analysis](http://web.cecs.pdx.edu/~fliu/papers/cvpr2012.pdf)
* Project: [http://web.cecs.pdx.edu/~fliu/](http://web.cecs.pdx.edu/~fliu/)
* Download: See [http://dpfan.net/d3netbenchmark/](http://dpfan.net/d3netbenchmark/)

SSB is also called STEREO dataset, which consists of 1000 pairs of binocular images.

#### LFSD

* Paper: [Saliency Detection on Light Field](https://ieeexplore.ieee.org/document/7570181)
* Project: [https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/](https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/)
* Download:
  * Official: See [https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/](https://sites.duke.edu/nianyi/publication/saliency-detection-on-light-field/)
  * See [http://dpfan.net/d3netbenchmark/](http://dpfan.net/d3netbenchmark/)

We acquire 100 light fields using the Lytro light field camera. For each light field, we provide: (a) Raw light field data, (b) A rough focal stack  (c) An all-focus image deriving from focal stack  (d) The ground truth corresponding to all-focus image.

To get a valid ground-truth, we ask three individuals to manually segment the saliency regions from the all-focus image. The result are deemed ground truth only when all three results are consistent (i.e., they have an overlap of over 90%)

#### RGBD135/DES

![image](assets/2019-05-23-10-44-38.png)

![depth](assets/2019-05-23-10-44-15.png)

![mask](assets/2019-05-23-10-44-59.png)

* Paper: [Depth Enhanced Saliency Detection Method](https://dl.acm.org/doi/pdf/10.1145/2632856.2632866)
* Project: [https://github.com/HzFu/DES_code](https://github.com/HzFu/DES_code)
* Download:

  * Official:
    + Baidu Pan: [https://pan.baidu.com/s/1pLv2B8n](https://pan.baidu.com/s/1pLv2B8n)
    + Google Drive: [https://onedrive.live.com/redir?resid=F3A8A31ABFAC51B0!256&amp;authkey=!AC4-yOEjn0bgrCQ&amp;ithint=file%2crar](https://onedrive.live.com/redir?resid=F3A8A31ABFAC51B0!256&authkey=!AC4-yOEjn0bgrCQ&ithint=file%2Crar)

  + See [http://dpfan.net/d3netbenchmark/](http://dpfan.net/d3netbenchmark/)

In our experiments, we provide a new RGB-D saliency detection dataset. We take 135 RGB-D indoor images by Kinect with the resolution 640×480. Then, three users are asked to mark the salient object of each image. We employ the overlapping areas of the manually labelled object as the ground truth.



### Resource Websites

* TC-11 Online Resources: [http://tc11.cvc.uab.es/datasets/type/](http://tc11.cvc.uab.es/datasets/type/)
* CVonline: Image Databases: [http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)
  + 中文: [https://blog.csdn.net/zhaoliang027/article/details/83376167](https://blog.csdn.net/zhaoliang027/article/details/83376167)
* MediaEval Benchmark: [http://www.multimediaeval.org/datasets/](http://www.multimediaeval.org/datasets/)
* Mit Saliency Benchmark: [http://saliency.mit.edu/datasets.html](http://saliency.mit.edu/datasets.html)
* Datasets for machine learning: [https://www.datasetlist.com/](https://www.datasetlist.com/)
* UCI machine learning repository: [https://archive.ics.uci.edu/ml/datasets.html](https://archive.ics.uci.edu/ml/datasets.html)
* Kaggle datasets: [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
* Google Dataset Seaerch:
  + [https://toolbox.google.com/datasetsearch](https://toolbox.google.com/datasetsearch)
  + [https://ai.google/tools/datasets/](https://ai.google/tools/datasets/)
  + [https://datasetsearch.research.google.com/](https://datasetsearch.research.google.com/)
  + AI开发者神器! 谷歌重磅推出数据集搜索 Dataset Search: [https://mp.weixin.qq.com/s/ErbwXAz-_AJrmUGMHZIcwg](https://mp.weixin.qq.com/s/ErbwXAz-_AJrmUGMHZIcwg)
  + Making it easier to discover datasets: [https://www.blog.google/products/search/making-it-easier-discover-datasets/](https://www.blog.google/products/search/making-it-easier-discover-datasets/)
* ⭐️⭐️⭐️ Yet Another Computer Vision Index To Datasets (YACVID): This website provides a list of frequently used computer vision datasets. Wait, there is more! There is also a description containing common problems, pitfalls and characteristics and now a searchable TAG cloud.: [http://yacvid.hayko.at/](http://yacvid.hayko.at/)
* ⭐️⭐️⭐️ IEEE DataPort provides a sustainable platform to all data owners in support of research and IEEE's overall mission of Advancing Technology for Humanity: [https://ieee-dataport.org/](https://ieee-dataport.org/)

