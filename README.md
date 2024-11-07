# LeFusion 3D Slicer Plugin

## Overview
This repository contains a plugin for 3D slicer that integrates the [LeFusion](https://github.com/M3DV/LeFusion/tree/main) model for 3D lung nodule inpainting in chest CT.

## Installation
- Create a virtual environment conda create -n lefusion python=3.10 and activate it conda activate lefusion
- Download the code:  git clone https://github.com/pedr0sorio/lefusion-slicer.git
- Check if your pip version is 22.3.1. If it is not, install pip version 22.3.1 pip install pip==22.3.1
- Enter the LeFusion folder cd lefusion-slicer/server and run pip install -r requirements.txt

```bash
conda create -n lefusion python=3.10 && \
conda activate lefusion && \
pip install pip==22.3.1 && \
cd lefusion-slicer/server && \
pip install -r requirements.txt
```

- Download the pre-trained LeFusion Model ([HuggingFace🤗](https://huggingface.co/YuheLiuu/LeFusion/tree/main/LIDC_LeFusion_Model))

   The LeFusion authors pre-trained LeFusion Model, which has been trained for 50,001 steps on the LIDC-IDRI dataset. This pre-trained model can be directly used for Inference if you do not want to re-train the LeFusion Model. Simply download it to `server/LeFusion_LIDC/LeFusion_model`.

   ```bash
   # Assuming cd is lefusion-slicer/server
   cd LeFusion_LIDC
   mkdir LeFusion_model
   cd LeFusion_model
   wget https://huggingface.co/YuheLiuu/LeFusion/resolve/main/LIDC_LeFusion_Model/model-50.pt -O model-50.pt
   ```

## Usage
Set model running on remote instance (or local machine if GPU is available), install plugin on local 3D slicer, add data to local slicer and inpaint images.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For any questions or issues, please open an issue on this repository or contact the maintainer at [pedro.c.osorio@gmail.com].

## ToDo List ✅ 🚀

✅ **ROI selection and volume editing in 3D slicer**  🚀

🔲 **Sending and retrieving the volumes from the server** 
- Make this repo run with functions and not scripts:
    - Refactor LeFusion inference script to a function with parameters that is called directly in the server script as opposed to creating a seperate script.

🔲 **Server side, processing, inference and return to client** 

🔲 **Update ReadMe**  

🔲 **Make opensource extension** 
