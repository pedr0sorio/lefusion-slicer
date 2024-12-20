# LeFusion 3D Slicer Plugin

<kbd>
<img src="media/ui.png">
</kbd>

## Overview
This repository contains a plugin for 3D slicer that integrates the [LeFusion](https://github.com/M3DV/LeFusion/tree/main) model for 3D lung nodule inpainting in chest CT.

It is composed of the code related to the slicer extension + the code related to the backend server where the inpainiting computations are done.

> [!NOTE]  
> You can have both the server and slicer running on the same local machine if you have a GPU connected to it. Otherwise, you can have the server running on a remote machine attached to a GPU and access it via port forwarding.

<kbd>
<img src="media/results-lfs.png">
</kbd>

## Installation

### Inpainting (LeFusion) Backend Setup

- Download the code: `git clone https://github.com/pedr0sorio/lefusion-slicer.git`
- Enter the LeFusion folder `cd lefusion-slicer/server`
- Create a virtual environment and activate it and install repo with poetry like so:

```bash
# assuming lefusion-slicer/server
conda env create -f environment.yml
conda activate lefusion
poetry config virtualenvs.create false
poetry install
```

#### Download the pre-trained LeFusion Model ([HuggingFace🤗](https://huggingface.co/YuheLiuu/LeFusion/tree/main/LIDC_LeFusion_Model))

   The LeFusion authors pre-trained LeFusion Model, which has been trained for 50,001 steps on the LIDC-IDRI dataset. This pre-trained model can be directly used for Inference. Simply download it to `server/lefusion/LeFusion_model`.

   ```bash
   # Assuming cd is lefusion-slicer/server
   cd lefusion
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
# from lefusion-slicer/server
python server.py
```

This runs the server on the public interface of your device on port 8888.

### 2. Basic plugin usage

<kbd>
<img src="media/under30-2.gif">
</kbd>

The notes present in the UI should explain how to use the inpainting tool. For visual support refer to the .gif above.

The Jump Length and Jump Number parameters control how many RePaint inference steps will be used to generate the lesions in the batch. Following the author's notes and also according to my experience using a value of 2 for both parameters offers the best compromise between lesion quality and compute time. 

On a A100 for default inference parameters the total processing time is approximately:

| Lesions in Batch | (1,1) | (2,2)    |
|------------------|-------|----------|
| 1                | 30s   | 1min     |
| 2                | 50s   | 1min 30s |
| 3                | -     | 2min 16s |
| 4                | -     | 2min 52s |

> [!NOTE]  
> I have been using a 40GB A100, which allows be to generate multiple lesions at once. Please adjust the batch size accoridng to you GPU. It is possible to run the server on a T4 GPU but batches would have to be reduced to a single lesion and teh total processing time soars up to 3min 30s / lesion. 

> [!NOTE]  
> As of now due to the lack of the DiffMask (see LeFusion paper) weights and code, the lesion mask for the selected crop will not be inferred but chosen randomly from a set of LIDC-IDRI masks already extracted by the original authors. They are saved under `server/in/mask-asset`.

## Contact
For any questions or issues, please open an issue on this repository or contact me at [pedro.c.osorio@gmail.com].

## Citation
If you are using this in your work, please cite both THIS REPO as defined in citation.cff and the original [LeFusion](https://github.com/M3DV/LeFusion/tree/main) paper.

## Acknowledgement

This repo is based on some code from [MedSAMSlicer](https://github.com/bowang-lab/MedSAMSlicer) and the inference code from LeFusion original authors [LeFusion](https://github.com/M3DV/LeFusion).

## ToDo List ✅ 

✅ **ROI selection and volume editing in 3D slicer**

✅ **Sending and retrieving the volumes from the server** 

✅ **Server side, processing, inference and return to client** 

🔲 **Add DiffMask to infer mask from crop** 

🔲 **Setting server running on Vertex AI inference endpoint**

🔲 **Further document codebase**  

🔲 **Update ReadMe**  

🔲 **Publish extension** 
