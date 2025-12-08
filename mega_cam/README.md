# Ocean-MEGA Lab Cam Dataset (OMEGA)
Videos manually spliced from 2023 highlight reel, available here:  
https://www.youtube.com/watch?v=qQF7RqVhywo

## How to Use
0. Follow DeepLabCut's (**DLC**)extensive [instructions](https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/installation.md) to install the toolkit on a local machine
1. Import the config.yaml file to set up a new project
2. Unzip all the labeled image files into its containing <ins>/labeled-data</ins> folder
3. Unzip the two files containing .mp4 clips into the <ins>/videos</ins> folder
4. Return to DLC's instructions, proceeding through GUI tabs + prompts to train + evaluate the network

## To-Do
- [ ] Convert DLC to YOLO dataset format  
   - e.g. via https://github.com/p-sira/deeplabcut2yolo/
- [ ] Evaluate dataset on Standalone Python instance
- [ ] Evaluate transfer learning capability of model from pre-trained COCO to OMEGA

![Fish23](https://github.com/user-attachments/assets/ff9317cd-9150-4b27-83e9-408201412a28)
