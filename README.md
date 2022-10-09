IN-MVAE Companion Code
======================

This repository contains the code for the paper "Multichannel Variational Autoencoder Based Speech Separa-tion in Designated Speaker Order" by [Lele Liao], [Guoliang Cheng], [Haoxin Ruan], [Kai Chen] and [Jing Lu].

Abstract
--------

The multichannel variational autoencoder (MVAE) integrates the rule-based update of separation matrix and the deep generative model and proves to be a competitive speech separation method. However, the output (global) permutation ambiguity still exists and turns out to be a fundamental problem in applications. In this paper, we address this problem by employing two dedicated en-coders. One encodes the speaker identity for the guidance of the output sorting, and the other en-codes the linguistic information for the reconstruction of the source signals. The instance normali-zation (IN) and the adaptive instance normalization (adaIN) are applied to the networks to dis-entangle the speaker representations from the content representations. The experimental results validate reliable sorting accuracy as well as good separation performance of the proposed method.



Test Run the Algorithms
-----------------------

The `main.py` programe in `code_vc` directory allows to train the IN-MVAE networks and test the different BSS algorithms.
