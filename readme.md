# TFGENZOO (Generative Model x Tensorflow 2.x)

![img](https://github.com/MokkeMeguru/TFGENZOO/workflows/tensorflow%20test/badge.svg?branch=master)
![img](https://img.shields.io/badge/License-MIT-yellow.svg)
![img](https://img.shields.io/badge/python-3.7-blue.svg)
![img](https://img.shields.io/badge/tensorflow-%3E%3D2.2.0-brightgreen.svg)
![img](https://badge.fury.io/py/TFGENZOO.svg)

# What&rsquo;s this repository?

This is a repository for some researcher to build some Generative models using Tensorflow 2.x.

I NEED YOUR HELP(please let me know about formula, implementation and anything you worried)

![img](https://raw.githubusercontent.com/MokkeMeguru/TFGENZOO/master/docs/tfgenzoo_header.png)

# Zen of this repository

    We don't want to need flexible architectures.
    We need strict definitions for shapes, parameters, and formulas.
    We should Implement correct codes with well-documented(tested).

# How to use?

## By Install

- pipenv

      pipenv install TFGENZOO==1.2.4.post7

- pip

      pip install TFGENZOO==1.2.4.post7

## Source build for development

1.  clone this repository (If you want to do it, I will push this repository to PYPI)
2.  build this repository `docker-compose build`
3.  run the environment `sh run_script.sh`
4.  connect it via VSCode or Emacs or vi or anything.

# Examples

- [TFGENZOO_EXAMPLE](https://github.com/MokkeMeguru/TFGENZOO_EXAMPLE)
- Simple Tutorials

  - [What is the invertible layer](./tutorials/01_What_is_the_invertible_layer.ipynb)

    The tutorial about Flow-based Model

  - [conditional flow-based model](./tutorials/02_conditional_flow-based_model.ipynb)

    How to add conditional input into Flow-based Model for the image generation.

# Documents

<https://mokkemeguru.github.io/TFGENZOO/>

# Roadmap

- [x] Flow-based Model Architecture (RealNVP, Glow)
- [ ] i-ResNet Model Architecture (i-ResNet, i-RevNet)
- [ ] GANs Model Architecture (GANs)

# Remarkable Backlog

Whole backlog is [here](https://github.com/MokkeMeguru/TFGENZOO/wiki/Backlog)

## News [2020/6/16]

New training results [Oxford-flower102](https://www.tensorflow.org/datasets/catalog/oxford_flowers102) with only 8 hours! (Quadro P6000 x 1)

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">data</th>
<th scope="col" class="org-right">NLL(test)</th>
<th scope="col" class="org-right">epoch</th>
<th scope="col" class="org-left">pretrained</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">Oxford-flower102</td>
<td class="org-right">4.590211391448975</td>
<td class="org-right">1024</td>
<td class="org-left">---</td>
</tr>
</tbody>
</table>

![img](https://raw.githubusercontent.com/MokkeMeguru/TFGENZOO/master/docs/oxford.png)

see more detail, you can see [my internship&rsquo;s report](https://docs.google.com/presentation/d/12z6MZizIsytLxUb2ly7vYorFiKruIGZ2ckQ0-By4b6s/edit?usp=sharing) (Japanese only, if you need translated version, please contact me.)

## News [2020/7/11]

Add some tutorial into `./tutorial`

## News [2021/3/30]

I wrote the master's paper about japanese text style transfer.  "AutoEncoder に基づく半教師あり和文スタ
イル変換"
https://drive.google.com/file/d/1KtkLZi6PUvL7msAqbg_KRdEC0pmmpbhf/view?usp=sharing

# Contact

MokkeMeguru ([@MokkeMeguru](https://twitter.com/MeguruMokke)): DM or Mention Please (in Any language).
