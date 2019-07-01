# Walsh-hadamard transform

Compressing images with a Hadamard transform

## Description

**From Wikipedia:** The Hadamard transform (also known as the *Walsh–Hadamard transform*, 
*Hadamard–Rademacher–Walsh transform*, *Walsh transform*, or *Walsh–Fourier transform*) is an example 
of a generalized class of Fourier transforms. It performs an orthogonal, symmetric, 
involutive, linear operation on 2m real numbers (or complex numbers, although the 
Hadamard matrices themselves are purely real).

The Hadamard transform can be regarded as being built out of *size-2 
discrete Fourier transforms* (DFTs), and is in fact equivalent to a 
multidimensional DFT of size `2 × 2 × ⋯ × 2 × 2`. It decomposes an 
arbitrary input vector into a superposition of *Walsh functions*.

The transform is named for the French mathematician Jacques Hadamard, 
the German-American mathematician Hans Rademacher, and the 
American mathematician Joseph L. Walsh. 

The Hadamard transform is also used in data encryption, as well as many signal processing 
and data compression algorithms, such as `JPEG XR` and `MPEG-4 AVC`. In video compression 
applications, it is usually used in the form of the sum of absolute transformed differences. 
It is also a crucial part of *Grover's algorithm* and *Shor's algorithm* in quantum computing. 

## Acknowledgement

This code is partially based on the solution from [ktisha/python2012](https://github.com/ktisha/python2012/tree/dee4beda8e22f3a66a3e31384d4b72ab66102e88/avereshchagin)

## Requirements

### Libraries

See file `requirements.txt` for a list of required packages

### Environment

Tested on `Windows` and `Python 2.7 (Anaconda2 build)`

## Effects

![https://raw.githubusercontent.com/oskar-j/walsh-hadamard-transform/master/doc/sample_usage.jpg](https://raw.githubusercontent.com/oskar-j/walsh-hadamard-transform/master/doc/sample_usage.jpg)
