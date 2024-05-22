
# [Real-time Topology Optimization in 3D via Deep Transfer Learning](https://www.sciencedirect.com/science/article/pii/S0010448521000257)

## Introduction

This project provides a graphical user interface (GUI) for [Real-time Topology Optimization in 3D via Deep Transfer Learning](https://www.sciencedirect.com/science/article/pii/S0010448521000257) that performs topology optimization using transfer learning models. The application allows users to select from pre-trained models and apply them to predict structures efficiently. The pre-trained models can be downloaded from [here.](https://drive.google.com/file/d/1n0tDb2HTGhG5BARF3XvqPw9Gvuosi9uX/view?usp=sharing)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Features

- **User-friendly GUI:** Built using PyQt5 for an intuitive user experience.
- **Pre-trained Models:** Includes multiple pre-trained models for topology optimization.
- **Visualization:** Utilizes Matplotlib and VTK for 2D and 3D visualizations.
- **Interactive Controls:** Provides sliders, buttons, and forms for easy interaction and customization.

## Installation

### Prerequisites

- Python 3.x
- PyQt5
- Keras
- NumPy
- Matplotlib
- SciPy
- VTK

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/MohammadBeh/TO-TL-GUI.git
   cd TO-TL-GUI
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the pre-trained models are in the same directory or update the paths in the script.

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Use the GUI to select the desired pre-trained model and configure the input parameters.

3. Visualize the results using the built-in plotting tools.

## Dependencies

- PyQt5
- Keras
- NumPy
- Matplotlib
- SciPy
- VTK

## Configuration

The paths to the pre-trained models are hardcoded in the script. Ensure the models are available in the specified paths or update the script accordingly.

## Examples

To visualize a sample optimization process:

1. Select the `TL` method.

2. Specify the dimension, domain, resolution, and boundary condition.

3. Click OK.

4. Change the force parameters to generate the structures interactively.

   Note: Changing the parameters (except force parameters) requires clicking on  `Reset` button. After that you can repeat the steps 1 to 4.

   Note 2: In the method, you can select `TL + SIMP` which runs a few iteration of SIMP solver on the predicted structure to obtain high quality and well connected structures (Only works for 2D structures).

## Troubleshooting

- Ensure all dependencies are installed and properly configured.
- Verify the paths to the pre-trained models.
- Check the console output for any error messages.

## Contributor

- Mohammad Behzadi - [MohammadBeh](https://github.com/MohammadBeh)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
