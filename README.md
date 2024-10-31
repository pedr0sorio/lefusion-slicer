# LeFusion 3D Slicer Plugin

## Overview
This repository contains a plugin for 3D slicer that integrates the LeFusion model for 3D lung nodule inpainting in chest CT.

## Installation
- Create a virtual environment conda create -n lefusion python=3.10 and activate it conda activate lefusion
- Download the code:  git clone https://github.com/pedr0sorio/lefusion-slicer.git
- Check if your pip version is 22.3.1. If it is not, install pip version 22.3.1 pip install pip==22.3.1
- Enter the LeFusion folder cd lefusion-slicer/server and run pip install -r requirements.txt

## Usage
Set model running on remote instance (Google cloud vertex is preferable!), install plugin on local 3D slicer, add data to local slicer and inpaint images.

## Contributing
Contributions are welcome! Please read the `CONTRIBUTING.md` for guidelines on how to contribute to this project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For any questions or issues, please open an issue on this repository or contact the maintainer at [pedro.c.osorio@gmail.com].
