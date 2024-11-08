# LeFusion 3D Slicer Plugin

<kbd>
<img src="media/under30-2.gif">
</kbd>

## Overview
This repository contains a plugin for 3D slicer that integrates the [LeFusion](https://github.com/M3DV/LeFusion/tree/main) model for 3D lung nodule inpainting in chest CT.

It is composed of the code related to the slicer extension + the code related to the backend server where the inpainiting computations are done.

**NOTE**: *You can have both the server and slicer running on the same local machine if you have a GPU connected to it. Otherwise, you can have the server running on a remote machine attached to a GPU and access it via port forwarding.*



## Installation

### Inpainting (LeFusion) Backend Setup

- Create a virtual environment and activate it: `conda create -n lefusion python=3.10` followed by `conda activate lefusion`
- Download the code: `git clone https://github.com/pedr0sorio/lefusion-slicer.git`
- Check if your pip version is 22.3.1. If it is not, install pip version 22.3.1: `pip install pip==22.3.1`
- Enter the LeFusion folder `cd lefusion-slicer/server` and run `pip install -r requirements.txt`

#### Equivalent full bash command:
```bash
conda create -n lefusion python=3.10
conda activate lefusion
pip install pip==22.3.1
cd lefusion-slicer/server
pip install -r requirements.txt
```

#### Download the pre-trained LeFusion Model ([HuggingFace🤗](https://huggingface.co/YuheLiuu/LeFusion/tree/main/LIDC_LeFusion_Model))

   The LeFusion authors pre-trained LeFusion Model, which has been trained for 50,001 steps on the LIDC-IDRI dataset. This pre-trained model can be directly used for Inference. Simply download it to `server/LeFusion_LIDC/LeFusion_model`.

   ```bash
   # Assuming cd is lefusion-slicer/server
   cd LeFusion_LIDC
   mkdir LeFusion_model
   cd LeFusion_model
   wget https://huggingface.co/YuheLiuu/LeFusion/resolve/main/LIDC_LeFusion_Model/model-50.pt -O model-50.pt
   ```

### Plugin Setup
- Install 3D Slicer from its official website. The compatibility of our plugin has been tested with 3D Slicer >= 5.6.2
- Select the Welcome to Slicer drop-down menu in the toolbar at the top and navigate to Developer Tools > Extension Wizard.
- Click on select Extension and locate the MedSAM2 folder under MedSAM2/slicer. Confirm if asked to import new module.
- Now, from the Welcome to Slicer drop-down menu, under the Segmentation sub-menu, MedSAM2 option is added. By choosing it, you can start using the plugin.

## Getting Started

### 1. Run the backend

You have to run the inpainting server to accept the incoming inpainting requests from slicer. You can do it both locally or on a remote computer:

```bash
# from repo root
python server/server.py
```

This runs the server on the public interface of your device on port 8888.

### 2. Basic plugin usage


## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For any questions or issues, please open an issue on this repository or contact me at [pedro.c.osorio@gmail.com].

## Acknowledgement

This repo is based on some code from [MedSAMSlicer](https://github.com/bowang-lab/MedSAMSlicer) and the inference code from LeFusion original authors [LeFusion](https://github.com/M3DV/LeFusion).

## ToDo List ✅ 🚀

✅ **ROI selection and volume editing in 3D slicer**  🚀

✅ **Sending and retrieving the volumes from the server** 

✅ **Server side, processing, inference and return to client** 

🔲 **Add DiffMask to infer mask from crop** 

🔲 **Setting server running on Vertex AI inference endpoint**

🔲 **Further document codebase**  

🔲 **Update ReadMe**  

🔲 **Publish extension** 
