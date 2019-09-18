# Readings for the ICDAR2019 Deep Learning Tutorial

## Original Convolutional Networks

- [1995-lecun-convolutional](General/1995-lecun-convolutional.pdf)
    - convolutional networks, sigmoid, average pooling
    - precursor of RCNN for multi-object recognition
    - digits and handwriting

## Convolutional Networks on GPUs

- [2013-krizhevsky-imagenet](General/2013-krizhevsky-imagenet.pdf)
    - ReLU, GPU training, local response normalization, pooling layers, dropout
    - Imagenet dataset
- [2014-srivastava-dropout](General/2014-srivastava-dropout.pdf)
    - dropouts as ensembles of networks
    - intended to prevent overtraining, improve generalization
    - standard test cases (CIFAR, MNIST, etc.)
- [2014-simonyan-maxpool-very-deep](General/2014-simonyan-maxpool-very-deep.pdf)
    - 19 weight layers, multicrop evaluation, "VGG team" ILSVRC-2014 challenge
- [2015-ioffe-batch-normalization](General/2015-ioffe-batch-normalization.pdf)
    - introduces batch normalization for faster training
- [2015-szegedy-rethinking-inception](General/2015-szegedy-rethinking-inception.pdf)
    - label smoothing, separable convolutions
- [2015-szegedy-going-deeper](General/2015-szegedy-going-deeper.pdf)
    - "inception modules", modular construction
- [2016-szegedy-inception](General/2016-szegedy-inception.pdf)
    - "inception modules", modular construction
- [2015-he-resnet](General/2015-he-resnet.pdf)
    - Introduces Resnet architecture
- [2015-jaderberg-spatial-transformer](General/2015-jaderberg-spatial-transformer.pdf)
    - adds spatial transformations/distortions to learnable primitives
- [2017-dai-deformable](General/2017-dai-deformable.pdf)
    - adds deformable convolutions to learnable primitives

OCR:

- [2013-breuel-high-performance-ocr-lstm](OCR/2013-breuel-high-performance-ocr-lstm.pdf)
    - LSTM for printed OCR
- [2013-goodfellow-multidigit](OCR/2013-goodfellow-multidigit.pdf)
    - Google SVHN digits, 200k numbers with bounding boxes
    - 8 layer convnet, ad-hoc sequence modeling
- [2017-breuel-lstm-ocr](OCR/2017-breuel-lstm-ocr.pdf)
    - comparison of different convnet+LSTM architectures for OCR

## Segmentation, Superresolution with Convolutional Networks

- [2015-dong-superresolution](General/2015-dong-superresolution.pdf)
    - explicit upscaling of images
- [2015-ronneberger-unet](General/2015-ronneberger-unet.pdf)
    - general U-net architecture for image-to-image mappings
- [2015-byeon-mdlstm-segmentation](General/2015-byeon-mdlstm-segmentation.pdf)
    - MDLSTM for image segmentation
- [2015-stollenga-pyramid-lstm](General/2015-stollenga-pyramid-lstm.pdf)
    - pyramid LSTM architecture
- [2015-long-convnet-semantic-segmentation](General/2015-long-convnet-semantic-segmentation.pdf)
    - semantic segmentation with convolutional networks
- [2015-girshick-rich-feature-hierarchies](General/2015-girshick-rich-feature-hierarchies.pdf)
    - semantic segmentation with convolutional networks (multitask)
- [2015-noh-deconvolutional-networks](General/2015-noh-deconvolutional-networks.pdf)
    - atrous convolutions
- [2017-blogpost-semantic-segmentation](Blogs/2017-blogpost-semantic-segmentation.pdf)
    - survey of semantic segmentation architectures
- [2016-chen-deeplab](General/2016-chen-deeplab.pdf)
- [2017-chen-deeplab-atrous](General/2017-chen-deeplab-atrous.pdf)
- [2017-chen-rethinking-atrous](General/2017-chen-rethinking-atrous.pdf)
    - atrous convolutions to learnable primitives, deeplab v3

OCR:

- [2015-afzal-binarization-mdlstm](OCR/2015-afzal-binarization-mdlstm.pdf)
    - MDLSTM for binarization (image-to-image transformation)
- [2017-breuel-mdlstm-layout](OCR/2017-breuel-mdlstm-layout.pdf)
    - layout analysis with MDLSTM
- [2017-chen-convnet-page-segmentation](OCR/2017-chen-convnet-page-segmentation.pdf)
    - layout analysis with convolutional nteworks
- [2017-he-semantic-page-segmentation](OCR/2017-he-semantic-page-segmentation.pdf)
    - layout analysis with convolutional nteworks
- [2018-mohan-layout-error-correction-using-dnn](OCR/2018-mohan-layout-error-correction-using-dnn.pdf)
    - layout analysis with convolutional nteworks

<!-- pix2pix etc. -->

## RCNN and Overfeat

- [2014-lecun-overfeat](General/2014-lecun-overfeat.pdf)
    - convolutional network, generic feature extraction
    - sliding window at multiple scales across image
    - regression network
- [2015-liu-multibox](General/2015-liu-multibox.pdf)
    - input image and ground truth boxes
- [2015-ren-faster-rcnn-v3](General/2015-ren-faster-rcnn-v3.pdf)
    - region proposal network (object/not object, box coords at each loc)
    - translation invariant anchors

OCR:

- [2014-jaderberg-convnet-ocr-wild](OCR/2014-jaderberg-convnet-ocr-wild.pdf)
    - convnet, R-CNN, bounding box regression
    - synthetic, ICDAR scene text, IIT Scene Text, IIT 5k words, IIT Sports-10k, BBC News
    - no bounding boxes in general; initial detector trained on positive word samples, negative images
    - 10k proposals per image

## Saliency, Attention, Visualization

- [2014-jiang-saliency](General/2014-jiang-saliency.pdf)
    - explicit computation of salience
- [2015-zhou-class-attention-mapping](General/2015-zhou-class-attention-mapping.pdf)
    - gradient-based mapping of class-related features
- [2016-selvaraju-gradient-mapping](General/2016-selvaraju-gradient-mapping.pdf)
    - gradient-based mapping of class-related features
- [2013-zeiler-visualizing-cnns](General/2013-zeiler-visualizing-cnns.pdf)
    - learns inverses to layers via unpooling, transposed convolutions
- [2016-yu-visualizing-vgg](General/2016-yu-visualizing-vgg.pdf)
    - applied to VGG16
- [2018-li-pyramid-attention](General/2018-li-pyramid-attention.pdf)
    - combines multiresolution and attention

## LSTM, CTC, GRU

- [1999-gers-lstm](General/1999-gers-lstm.pdf)
    - introduces the LSTM architecture
- [2005-graves-bdlstm](General/2005-graves-bdlstm.pdf)
    - introduces bidirectional LSTM
- [2006-graves-ctc](General/2006-graves-ctc.pdf)
    - introduces CTC alignment (a kind of forward-backward algorithm)

OCR:

- [2012-elaguni-ocr-in-video](OCR/2012-elaguni-ocr-in-video.pdf)
    - manually labeled training data on small dataset
    - multiscale, convnet features, BLSTM, CTC
- [2014-bluche-comparison-sequence-trained](OCR/2014-bluche-comparison-sequence-trained.pdf)
    - HMM, GMM-HMM, MLP-HMM, LSTM
    - Rimes, IAM; decoding with Kaldi (ASR toolkit)
- [2016-he-reading-scene-text](OCR/2016-he-reading-scene-text.pdf)
    - large CNN, Maxout units, LSTM, CTC
    - Street View Text, IIT 5k-word, PhotoOCR, etc., using bounding boxes for training
- [2017-wang-gru-ocr](OCR/2017-wang-gru-ocr.pdf)

## 2D LSTM

- [2009-graves-multidimensional](General/2009-graves-multidimensional.pdf)
    - applies LSTM to multidimensional problems
- [2014-byeon-supervised-texture](General/2014-byeon-supervised-texture.pdf)
    - supervised image segmentation using multidimensional LSTM
- [2016-visin-reseg](General/2016-visin-reseg.pdf)
    - separable multidimensional LSTMs for image segmentation
- [2015-sonderby-convolutional](General/2015-sonderby-convolutional.pdf)
    - convolutional LSTM architecture and attention
- [2016-shi-convolutional-lstm](General/2016-shi-convolutional-lstm.pdf)
    - convolutional LSTM architecture

OCR:

- [2015-visin-renet](OCR/2015-visin-renet.pdf)
    - separable multidimensional LSTMs for OCR

## Seq2Seq, Attention

- [2012-graves-sequence-transduction](General/2012-graves-sequence-transduction.pdf)
    - introduces sequence transduction as an alternative to CTC
- [2015-bahdanau-attention](General/2015-bahdanau-attention.pdf)
    - content-based attention mechanisms for sequence to sequence tasks
- [2015-zhang-character-level-convnets-text](General/2015-zhang-character-level-convnets-text.pdf)
    - simple use of convolutional networks as alternatives to n-grams, sequence models
- [2016-chorowski-better-decoding](General/2016-chorowski-better-decoding.pdf)
    - label smoothing and beam search
- [2017-vaswani-attention-is-all-you-need](General/2017-vaswani-attention-is-all-you-need.pdf)
    - high performance sequence-to-sequence with attention
    - masked, multi-head attention
- [2017-prabhavalkar-s2s-comparison](General/2017-prabhavalkar-s2s-comparison.pdf)
    - a comparison of different sequence-to-sequence approaches
- [2017-gehring-convolutional-s2s](General/2017-gehring-convolutional-s2s.pdf)
    - purely convolutional sequence-to-sequence with attention

OCR:

- [2015-sahu-s2s-ocr](OCR/2015-sahu-s2s-ocr.pdf)
    - standard seq2seq encoder/decoder approach
    - TSNE visualizations of encoded word images
    - word images from scanned books

## Visual Attention

- [2017-nam-dual-attention](General/2017-nam-dual-attention.pdf)
    - joint visual and text attention networks

OCR:

- [2016-bluche-end-to-end-hw-mdlstm-attention](OCR/2016-bluche-end-to-end-hw-mdlstm-attention.pdf)
    - full paragraph handwriting recognition without explicit segmentation
    - MDLSTM plus attention, tracking, etc.
    - IAM database, pretraining LSTM+CTC, curriculum learning
- [2016-lee-recursive-recurrent-attention-wild](OCR/2016-lee-recursive-recurrent-attention-wild.pdf)
    - recursive convolutional layers, tied weights, followed by attention, character level modeling
    - ICDAR 2003, 2013, SVT, IIT5k, Synth90k using bounding boxes for training

## Language Modeling

- [2016-rosca-lstm-transcript](OCR/2016-rosca-lstm-transcript.pdf)

## Domain Adaptation, Unsupervised, Semi-Supervised, Multitask Learning

Domain Adaptation:

- [2017-liu-unsupervised-domain-adaptation](Learning/2017-liu-unsupervised-domain-adaptation.pdf)
- [2017-tzen-adversarial-domain-discriminator-adaptation](Learning/2017-tzen-adversarial-domain-discriminator-adaptation.pdf)

Semi-Supervised Learning:

- [2005-zhu-semi-supervised](Learning/2005-zhu-semi-supervised.pdf)
    - classical methods of semi-supervised learning
- [2017-li-noisy-labels-distillation](Learning/2017-li-noisy-labels-distillation.pdf)
    - uses distillation for dealing with noisy lables
- [2018-oliver-evaluation-semi-supervised](Learning/2018-oliver-evaluation-semi-supervised.pdf)
- [2018-ren-metalearning-semi-supervised](Learning/2018-ren-metalearning-semi-supervised.pdf)
- [2018-tanaka-joint-optimization-noisy-labels](Learning/2018-tanaka-joint-optimization-noisy-labels.pdf)

Examples of Unsupervised Learning:

- [2016-lin-unsupervised-binary-descriptors](Learning/2016-lin-unsupervised-binary-descriptors.pdf)
- [2016-radford-unsupervised-representation-learning](Learning/2016-radford-unsupervised-representation-learning.pdf)
- [2016-xie-unsupervised-deep-embedding](Learning/2016-xie-unsupervised-deep-embedding.pdf)
- [2017-ren-unsupervised-deep-flow](Learning/2017-ren-unsupervised-deep-flow.pdf)
- [2017-lotter-unsupervised-predictive-video-coding](Learning/2017-lotter-unsupervised-predictive-video-coding.pdf)
- [2018-li-unsupervised-odometry](Learning/2018-li-unsupervised-odometry.pdf)

Transfer and Multitask Learning:

- [2016-geng-transfer-learning-reid](Learning/2016-geng-transfer-learning-reid.pdf)
- [2016-rusu-progressive-networks](Learning/2016-rusu-progressive-networks.pdf)
- [2017-ruder-multitask-survey](Learning/2017-ruder-multitask-survey.pdf)

## GANs

- [2014-goodfellow-gans](General/2014-goodfellow-gans.pdf)
- [2015-radford-dcgan](General/2015-radford-dcgan.pdf)
- [2016-isola-image2image-gan](General/2016-isola-image2image-gan.pdf)
- [2016-salimans-improved-gan-training](General/2016-salimans-improved-gan-training.pdf)

<!-- - [2016-ho-gan-imitation-learning](General/2016-ho-gan-imitation-learning.pdf) -->
<!-- # Siamese -->

## Computational Issues

- [2017-chen-distributed-sgd](General/2017-chen-distributed-sgd.pdf)
    - scaling up with many GPUs on many nodes
- [2016-iandola-squeezenet](General/2016-iandola-squeezenet.pdf)
    - reducing the number of parameters
- [2017-yuan-adversarial](General/2017-yuan-adversarial.pdf)
    - the existence and problems with adversarial samples

# Surveys

- [2014-schmidhuber-deep-learning-survey](General/2014-schmidhuber-deep-learning-survey.pdf)
- [2015-lecun-nature-deep-learning](General/2015-lecun-nature-deep-learning.pdf)
- [2015-karpathy-recurrent-ocr](Blogs/2015-karpathy-recurrent-ocr.pdf)
- [2018-alom-survey-imagenet](General/2018-alom-survey-imagenet.pdf)

# Additional Readings

- [2013-goodfellow-maxout](More/2013-goodfellow-maxout.pdf)
- [2014-donahue-long-term-rcnn](General/2014-donahue-long-term-rcnn.pdf)
- [2014-karpathy-image-descriptions](General/2014-karpathy-image-descriptions.pdf)
- [2015-liu-face-attributes-wild](General/2015-liu-face-attributes-wild.pdf)
- [2015-mnih-deep-reinforcement-learning](General/2015-mnih-deep-reinforcement-learning.pdf)
- [2015-ng-video-classification](General/2015-ng-video-classification.pdf)
- [2015-yu-visual-madlibs](General/2015-yu-visual-madlibs.pdf)
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

