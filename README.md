# Interactive Foreground Extraction using Iterated Graph Cuts [GrabCut]

This repository contains code for the paper [Interactive Foreground Extraction using Iterated Graph Cuts](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf) implemented as a project submission for the `Statistical Methods in AI(SMAI)` course offered in Monsoon 2021.

__Team Name__: Outliers

**Team Members**:
- Sai Amrit Patnaik 
- Sabyasachi Mukhopadhyay
- Neha S.
- Mehul Arora 



__Assigned TA__ : Abhinaba Bala

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

The `main.py` script in the repository is the main script which can be used to run the demo. The same code is also in the `demo.ipynb` notebook and the full working notebook with all the code at one place is in the `notebooks/Grabcut Full.ipynb`. 

The program accepts command line parameters such as the example given below:

```
python main.py --image_path <path to input image>
```
The above command takes the input image from the specified path and returns a segmented output on the output screen.

The instructions to run the script is:
1. Run the above command.
2. The interactive UI is generated which has the input image which is to be segmented.
3. Press the Right mouse button and drag over the screen over the section of the image to mark as foreground.
4. After drawing the bounding box, release the mouse button and press `Enter` key to start the segmentation.
5. The output will be generated and displayed on the output screen in a few seconds.

#### For Interactive Improvement

- Press `0` to mark as `background` or `1` to mark as `foreground` and press `Enter` again to update the result

For more information on the possible parameters, we can also check the help flag as follows:
```
python main.py --help

usage: main.py [-h] [--input_path INPUT_PATH] [--n_iters N_ITERS]
               [--gamma GAMMA] [--gmm_components GMM_COMPONENTS]
               [--neighbours NEIGHBOURS] [--color_space COLOR_SPACE]

Grabcut Segmentation parameters Parser.
Pass the parameters following instructions given below to run the demo experiment.

optional arguments:
  -h, --help                       :  show this help message and exit
  --input_path INPUT_PATH          :  Path to image for processing
  --n_iters N_ITERS                :  No of iterations to run
  --gamma GAMMA                    :  Value for Gamma parameter
  --gmm_components GMM_COMPONENTS  :  Number of GMM components for each GMM
  --neighbours NEIGHBOURS          :  8 or 4 connectivity to be considered
  --color_space COLOR_SPACE        :  Parameter value for beta

```
As shown above, the parameters listed in the help menu can be altered via the command line. We show some ablation study over the various parameters like:
1. Number of Iterations
2. Number of GMM components
3. The Gamma Parameter
4. 4 or 8 neighbours connectivity
5. Color Space of the Input Image
6. Size of the initial bounding box

We present the results and reasonings behind the demonstrated behaviour by the method across the various parameters. 

For a more detailed report and analysis, we recommend to look into the report [X](./documents/report.pdf) presentation [here](./documents/mideval.pdf), also available [here](https://docs.google.com/presentation/d/19GnXUvxiNn3NGnubRpFsgUwSo3tyaixqB04HjuzGzyM/edit?usp=sharing).
