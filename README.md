# Readings for the ICDAR2019 Deep Learning Tutorial

## Original Convolutional Networks

- [1995-lecun-convolutional](General/1995-lecun-convolutional.pdf)

## Convolutional Networks on GPUs

- [2013-krizhevsky-imagenet](General/2013-krizhevsky-imagenet.pdf)

## RCNN

- [2013-girshick-rcnn](General/2013-girshick-rcnn.pdf)
- [2015-girshick-fast-rcnn](General/2015-girshick-fast-rcnn.pdf)
- [2015-ren-faster-rcnn](General/2015-ren-faster-rcnn.pdf)
- [2015-ren-faster-rcnn-v3](General/2015-ren-faster-rcnn-v3.pdf)

## Visualizing

- [2013-zeiler-visualizing-cnns](General/2013-zeiler-visualizing-cnns.pdf)
- [2016-yu-visualizing-vgg](General/2016-yu-visualizing-vgg.pdf)

## ReLU, MaxPool, Dropout

- [2014-srivastava-dropout](General/2014-srivastava-dropout.pdf)
- [2014-lecun-overfeat](General/2014-lecun-overfeat.pdf)
- [2014-simonyan-maxpool-very-deep](General/2014-simonyan-maxpool-very-deep.pdf)
- [2015-szegedy-going-deeper](General/2015-szegedy-going-deeper.pdf)

## Superresolution

- [2015-dong-superresolution](General/2015-dong-superresolution.pdf)

## GANs

- [2014-goodfellow-gans](General/2014-goodfellow-gans.pdf)
- [2015-radford-dcgan](General/2015-radford-dcgan.pdf)

<!-- # Siamese -->

## Saliency

- [2014-jiang-saliency](General/2014-jiang-saliency.pdf)

## Batchnorm

- [2015-ioffe-batch-normalization](General/2015-ioffe-batch-normalization.pdf)

## UNET

- [2015-ronnenberger-unet](General/2015-ronnenberger-unet.pdf)

## Resnet

- [2015-he-resnet](General/2015-he-resnet.pdf)
- [2016-szegedy-inception](General/2016-szegedy-inception.pdf)

## Strided and Atrous Convolutions

- [2017-chen-deeplab-atrous](General/2017-chen-deeplab-atrous.pdf)
- [2017-chen-rethinking-atrous](General/2017-chen-rethinking-atrous.pdf)

## Transformer Networks

- [2015-jaderberg-spatial-transformer](General/2015-jaderberg-spatial-transformer.pdf)

## LSTM, CTC, GRU

- [1999-gers-lstm](General/1999-gers-lstm.pdf)
- [2005-graves-bdlstm](General/2005-graves-bdlstm.pdf)
- [2006-graves-ctc](General/2006-graves-ctc.pdf)

## 2D LSTM

- [2009-graves-multidimensional](General/2009-graves-multidimensional.pdf)
- [2014-byeon-supervised-texture](General/2014-byeon-supervised-texture.pdf)
- [2016-visin-reseg](General/2016-visin-reseg.pdf)
- [2015-visin-renet](OCR/2015-visin-renet.pdf)

## Seq2Seq, Attention

- [2012-graves-sequence-transduction](General/2012-graves-sequence-transduction.pdf)
- [2016-chorowski-better-decoding](General/2016-chorowski-better-decoding.pdf)
- [2015-bahdanau-attention](General/2015-bahdanau-attention.pdf)
- [2017-vaswani-attention-is-all-you-need](General/2017-vaswani-attention-is-all-you-need.pdf)
- [2017-prabhavalkar-s2s-comparison](General/2017-prabhavalkar-s2s-comparison.pdf)

## Visual Attention

- [2017-nam-dual-attention](General/2017-nam-dual-attention.pdf)
- [2016-you-dual-attention](General/2016-you-dual-attention.pdf)

## Deformable Convolutions

- [2017-dai-deformable](General/2017-dai-deformable.pdf)

## Parallel and Distributed Training

- [2017-chen-distributed-sgd](General/2017-chen-distributed-sgd.pdf)

## Adversarial Samples

- [2017-yuan-adversarial](General/2017-yuan-adversarial.pdf)

## Squeezing

- [2016-iandola-squeezenet](General/2016-iandola-squeezenet.pdf)

## Surveys

- [2014-schmidhuber-deep-learning-survey](General/2014-schmidhuber-deep-learning-survey.pdf)
- [2015-lecun-nature-deep-learning](General/2015-lecun-nature-deep-learning.pdf)
- [2015-karpathy-recurrent-ocr](OCR/2015-karpathy-recurrent-ocr.pdf)
- [2018-alom-survey-imagenet](General/2018-alom-survey-imagenet.pdf)

## OCR

- [2012-elaguni-ocr-in-video](OCR/2012-elaguni-ocr-in-video.pdf)
    - manually labeled training data on small dataset
    - multiscale, convnet features, BLSTM, CTC
- [2013-goodfellow-multidigit](OCR/2013-goodfellow-multidigit.pdf)
    - Google SVHN digits, 200k numbers with bounding boxes
    - 8 layer convnet, ad-hoc sequence modeling
- [2014-bluche-comparison-sequence-trained](OCR/2014-bluche-comparison-sequence-trained.pdf)
    - HMM, GMM-HMM, MLP-HMM, LSTM
    - Rimes, IAM; decoding with Kaldi (ASR toolkit)
- [2014-jaderberg-convnet-ocr-wild](OCR/2014-jaderberg-convnet-ocr-wild.pdf)
    - convnet, R-CNN, bounding box regression
    - synthetic, ICDAR scene text, IIT Scene Text, IIT 5k words, IIT Sports-10k, BBC News
    - no bounding boxes in general; initial detector trained on positive word samples, negative images
    - 10k proposals per image
- [2015-sahu-s2s-ocr](OCR/2015-sahu-s2s-ocr.pdf)
    - standard seq2seq encoder/decoder approach
    - TSNE visualizations of encoded word images
    - word images from scanned books
- [2016-bluche-end-to-end-hw-mdlstm-attention](OCR/2016-bluche-end-to-end-hw-mdlstm-attention.pdf)
    - full paragraph handwriting recognition without explicit segmentation
    - MDLSTM plus attention, tracking, etc.
    - IAM database, pretraining LSTM+CTC, curriculum learning
- [2016-he-reading-scene-text](OCR/2016-he-reading-scene-text.pdf)
    - large CNN, Maxout units, LSTM, CTC
    - Street View Text, IIT 5k-word, PhotoOCR, etc., using bounding boxes for training
- [2016-lee-recursive-recurrent-attention-wild](OCR/2016-lee-recursive-recurrent-attention-wild.pdf)
    - recursive convolutional layers, tied weights, followed by attention, character level modeling
    - ICDAR 2003, 2013, SVT, IIT5k, Synth90k using bounding boxes for training

## Additional Readings

- [2013-goodfellow-maxout](More/2013-goodfellow-maxout.pdf)
- [2014-donahue-long-term-rcnn](General/2014-donahue-long-term-rcnn.pdf)
- [2014-karpathy-image-descriptions](General/2014-karpathy-image-descriptions.pdf)
- [2015-liu-face-attributes-wild](General/2015-liu-face-attributes-wild.pdf)
- [2015-mnih-deep-reinforcement-learning](General/2015-mnih-deep-reinforcement-learning.pdf)
- [2015-ng-video-classification](General/2015-ng-video-classification.pdf)
- [2015-ronneberger-unet](General/2015-ronneberger-unet.pdf)
- [2015-yu-visual-madlibs](General/2015-yu-visual-madlibs.pdf)
- [2015-zhang-character-level-convnets-text](General/2015-zhang-character-level-convnets-text.pdf)
- [2015-zheng-crfs-as-rnns](General/2015-zheng-crfs-as-rnns.pdf)
- [2016-abadi-tensorflow](General/2016-abadi-tensorflow.pdf)
- [2016-ba-layernorm](More/2016-ba-layernorm.pdf)
- [2016-mnih-async-dl](General/2016-mnih-async-dl.pdf)
- [2016-salimans-weightnorm](More/2016-salimans-weightnorm.pdf)
- [2016-shi-superresolution](More/2016-shi-superresolution.pdf)
- [2016-ulyanof-instancenorm](More/2016-ulyanof-instancenorm.pdf)
- [2016-vinyals-matching-networks](General/2016-vinyals-matching-networks.pdf)
- [2016-zhang-very-deep-speech](General/2016-zhang-very-deep-speech.pdf)
- [2017-barron-celu](More/2017-barron-celu.pdf)
- [2017-hochrreiter-self-normalizing-networks](More/2017-hochrreiter-self-normalizing-networks.pdf)
- [2017-ioffe-batchnorm](More/2017-ioffe-batchnorm.pdf)
- [2017-wang-tacotron](General/2017-wang-tacotron.pdf)
- [2018-burda-curiosity](General/2018-burda-curiosity.pdf)
- [2018-metz-metalearning](General/2018-metz-metalearning.pdf)
- [2018-wu-groupnorm](More/2018-wu-groupnorm.pdf)
