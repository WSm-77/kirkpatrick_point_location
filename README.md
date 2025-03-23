# Kirkpatrick Point Location Algorithm

This repository serves as an educational resource explaining the working principles of the Kirkpatrick point location algorithm.

## Showcase

<img src="description_resources/kikrpatrick_point_location.gif" alt="Kirkpatrick point location algorithm showcase">

## Cloning the Repository

To clone this repository, use the following command:

```bash
git clone https://github.com/WSm-77/Kirkpatrick-point-location.git
```

## bit\_algo\_vis\_tool Utility

During the project, we utilized a tool provided by the ***Bit*** student's research group.

### Setting Up the Environment

To properly configure the environment, follow these steps:

#### 1. Create a Virtual Environment

```bash
conda create --name kirkpatrick python=3.9
conda activate kirkpatrick
```

#### 2. Install Required Packages

```bash
python3 setup.py sdist
python3 -m pip install -e .
```

#### 3. Usage

Once inside the newly created environment (you should see **(kirkpatrick)** before the username in the terminal), create a Jupyter notebook and select the Python interpreter from this environment as the kernel.


> [!TIP]
> For standard Python scripts (*.py* files) in VSCode, you can set the interpreter to the one from the **kirkpatrick** environment:
>
> ```Ctrl+Shift+P > Python: Select Interpreter```

The location where you create new files does not matter. The environment configuration ensures that the *bit\_algo\_vis\_tool* folder is accessible from anywhere in the project, just like any other Python package such as *numpy* or *matplotlib*. For example, to use the ***Visualizer*** class, simply import it:

```python
from bit_algo_vis_tool.visualizer.visualizer import Visualizer
```

If you encounter issues, the first troubleshooting step should be restarting the environment:

```bash
conda deactivate
conda activate kirkpatrick
```
