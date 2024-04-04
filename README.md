# LidarManning - Measuring Manning's n from Point Clouds for Hydrodynamic Flood Models [![DOI](https://zenodo.org/badge/779379669.svg)](https://zenodo.org/doi/10.5281/zenodo.10913207)
This repository contains the necessary code to apply a laboratory-trained Deep Neural Network to measure Manning's n based on point clouds. Details of this research are available in [this preprint](https://arxiv.org/abs/2404.02234). 

```
Haces-Garcia, F., Kotzamanis, V., Glennie, C., & Rifai, H. (2024). Deep Neural Networks with 3D Point Clouds for Empirical Friction Measurements in Hydrodynamic Flood Models. Retrieved from http://arxiv.org/abs/2404.02234. Submitted on 04/03/2024 for review at Water Resources Research.
```

## Installation

To get started, follow these steps:

### 1. Install Anaconda

First, you need to install Anaconda, which is a free and open-source distribution of Python and R programming languages for scientific computing, that aims to simplify package management and deployment. You can download and install Anaconda from the [official website](https://www.anaconda.com/products/distribution).

### 2. Clone the Repository

Clone this repository to your local machine and navigate to it:

```bash
git clone https://github.com/f-haces/LidarManning.git
cd LidarManning
```

### 3. Create and Activate Conda Environment 
```bash
conda env create -f environment.yml
conda activate lidarmanning
```

## Code Structure
The code in this repository is structured as follows: 

* [ProcessRasters.ipynb](ProcessRasters.ipynb): Jupyter Notebook showing how to use the code in [PointUtils.py](PointUtils.py) to directly read a [LasPy-compatible](https://laspy.readthedocs.io/en/latest/index.html) Point Cloud and obtain a GeoTIFF raster of Manning's n. 
* [CompositeManningsN.ipynb](CompositeManningsN.ipynb): Jupyter Notebook showing how to apply the results from [ProcessRasters.ipynb](ProcessRasters.ipynb) to a HEC-RAS 1D model based on Cross Sections using the tools in [rasUtils.py](rasUtils.py). 
* [MRNN_012024.pth](MRNN_012024.pth): Latest model weights for [PointNet adaptation](PointNet.py). 

## Contributing
We welcome contributions from the community! We're excited to invite developers and modelers to test this software with their own point clouds and study areas. We appreciate all feedback! 

 Whether you're a seasoned developer or just starting out, there are many ways you can contribute. Feel free to engage with us by asking questions, providing feedback, reporting bugs, or suggesting new features. If you encounter any issues, please don't hesitate to raise an issue on GitHub. Additionally, if you have any ideas for enhancements or would like to contribute code, we encourage you to fork the repository, make your changes, and submit a pull request.

## License
This code is distributed through an [MIT License](LICENSE).



