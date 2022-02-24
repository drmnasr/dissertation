# dissertation
## Folder Structure:
    - Models : Contains all the trained models, class names and colors.
    - Datasets: Contains a script to remap ADE20K

## Download Model Files:
https://drive.google.com/drive/folders/1i313ubfDbe71gf_yzfmhZHmilnByQZEt?usp=sharing

## Requirements:
Will only run on Jetson devices
Install Jetson Inference repo at https://github.com/dusty-nv/jetson-inference first

## Run
Run: python3 segnet.py --model *choose model path* --width=960 --height=720  --input_blob=input_0 --output_blob=output_0 --labels=classes.txt --colors=colors.txt
