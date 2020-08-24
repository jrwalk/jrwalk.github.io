---
title: Nickname Generation with Recurrent Neural Networks with PyTorch
layout: post_norel
permalink: /pages/projects/nickname-generation-with-pytorch
---

tl;dr [read more on the Pluralsight tech blog](https://www.pluralsight.com/tech-blog/nickname-generation-with-recurrent-neural-networks-with-pytorch/)

Anyone who’s attended one of the PAX gaming conventions has encountered a group called (somewhat tongue-in-cheek) the “Enforcers”.
This community-driven group handles much of the operations of the convention, including anything from organizing queues for activities (and entertaining people in them) to managing libraries of games or consoles for checkout.
Traditionally, Enforcers will often use an old-school forum handle during the convention, with the pragmatic outcome of easily distinguishing between Enforcers with common names, but also bringing some personal style – some even prefer to be addressed by handle over their real name during a PAX.
Handles like Waterbeards, Gundabad, Durinthal, bluebombardier, Dandelion, ETSD, Avogabro, or IaMaTrEe will draw variously from ordinary names, puns, personal background, or literary/pop-culture references.

At PAX East this past February, we hit upon the idea of building a model to generate new Enforcer handles.
This a perfect use case for a sequential model, like a recurrent neural network – by training on a sequence of data, the model learns to generate new sequences for a given seed (in this case, predicting each successive character based on the word so far).

I recently published a post on [Pluralsight's tech blog](https://www.pluralsight.com/tech-blog/nickname-generation-with-recurrent-neural-networks-with-pytorch/) on building recurrent neural networks with [PyTorch](https://pytorch.org/).
My goal was twofold:

- to explore building a recurrent neural network in PyTorch (which is new ground for me, as most of our production implementations are in Tensorflow instead)
- to demonstrate the power of transfer learning.

Some of my favorite generated handles:

```
seed --> output
---------------
a --> andelian
s --> saminthan
C --> Carili
b --> bestry
E --> EliTamit
G --> Gusty
L --> Landedbandareran
M --> Muldobban
I --> Imameson
T --> Thichy
G --> Gato
b --> boderton
K --> KardyBan
c --> cante
s --> sandestar
o --> oldarile
L --> Laulittyy
P --> Parielon
X --> Xilly
E --> Elburdelink
Be --> Beardog
Ew --> EwToon
Zz --> Zzandola
Ex --> Exi-Hard
Sp --> Spisfy
Sm --> Smandar
Yr --> Yrian
Fou --> Foush
Qwi --> Qwilly
Ixy --> IxyBardson
Abl --> Ablomalad
Ehn --> Ehnlach
Ooz --> Oozerntoon
```

Enjoy!
