# Installation

1. Clone the GroundingDINO repository from GitHub.

```bash
git clone https://github.com/pokemon12332112/Zero_shot_counting_with_Dino.git
```
- Create virtual environment: 
```bash
python -m venv lcount
lcount/Scripts/activate
pip install matplotlib opencv-python notebook tqdm
pip install torch torchvision
```
2. Change the current directory to the GroundingDINO folder.
```bash
cd GroundingDINO/
```
3. Install the required dependencies in the current directory.
```bash
pip install -e .
```
4. Download pre-trained model weights.
```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
# Download dataset
- Images can be downloaded from here: [This link](https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing)

- Precomputed density maps can be found here: [This link](https://archive.org/details/FSC147-GT)

- Place the unzipped image directory and density map directory inside the data directory.
# How to run the code
1. Quick demo
```bash
python main_demo.py --input-image path/to/your/image
```
2. Evaluation
- Testing on validation set
```bash
python test_with_GD.py --data_path /PATH/TO/YOUR/FSC147/DATASET/ --test_split val
```
- Testing on test set
```bash
python test_with_GD.py --data_path /PATH/TO/YOUR/FSC147/DATASET/ --test_split test
```
# License

[MIT](https://choosealicense.com/licenses/mit/)
