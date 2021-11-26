# Interactive Foreground Extraction using Iterated Graph Cuts [GrabCut]

This repository contains code for the paper [Interactive Foreground Extraction using Iterated Graph Cuts](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf) implemented as a project submission for the `Statistical Methods in AI(SMAI)` course offered in Monsoon 2021.

__Team Name__: Outliers

**Team Members**:
- Sai Amrit Patnaik 
- Sabyasachi Mukhopadhyay
- Neha S.
- Mehul Arora 
- git 



__Assigned TA__  : Abhinaba Bala

# GrabCut
GrabCut is an algorithm for image segmentation based on graph cuts. It is used to extract the foreground objects in images with minimal user interaction. Initially, the user needs to draw a bounding-box (i.e., a rectangle) around the foreground object to be segmented. Based on these regions, the color distribution of the foreground and the background objects are estimated by the algorithm using Gaussian mixture model. This is then followed by the iterative energy minimization process to get the best estimates and segment the foreground object.

![](https://github.com/saiamrit/grabcut-smai/blob/main/imgs/grabcut.jpg)


The following sections outline how to run the demo and some examples of the expected output from running the mentioned scripts.

## Code Structure

The code is structured as follows:

```
.
├── README.md
├── documents
│   ├── Mid-eval.pdf
│   └── proposal.pdf
├── imgs
|── main.py
└── src
    ├── __init__.py
    ├── gmm.py
    ├── ui.py
    └── grabcut.py
```
In the above structure, the source code for the whole implementation can be found in the `src` directory. The scripts each contain a description of the functions/classes implemented and provide a wrapper to experiment with the flow of the program. The `main.py` accesses all these functionality provided in the `src` library.

## Run Demo

### Pre-requisites

Before running the demo, make sure that the following python libraries are installed and working well. The code available in this repository makes use of the following to function completely.

```
numpy
opencv-python
scikit-learn
igraph
```
### Data

The sample images on which we ran our code can be found in the `imgs` folder. These images are stock images downloaded from Google.

### Running the demo

The `main.py` script in the repository is the main script which can be used to run the demo. The instructions to run the script is:
1. The interactive UI is generated which has the input image which is to be segmented.
2. Press the Left mouse button and drag over the screen over the section of the image to mark as foreground.
3. After drawing the bounding box, release the mouse button and press `Enter` key to start the segmentation.
4. The output will be generated and displayed on the output screen in a few seconds.

For a more detailed report and analysis, we recommend to look into the presentation [here](./documents/mideval.pdf), also available [here](https://docs.google.com/presentation/d/19GnXUvxiNn3NGnubRpFsgUwSo3tyaixqB04HjuzGzyM/edit?usp=sharing).
