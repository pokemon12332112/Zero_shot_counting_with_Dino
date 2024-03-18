"""
Demo file for Few Shot Counting

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 19-Apr-2021
Last modified: 19-Apr-2021
"""

import cv2
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import visualize_output_and_save, select_exemplar_rois
from PIL import Image
import os
import torch
import argparse
import torch.optim as optim
from utils import MincountLoss, PerturbationLoss
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision
from torchvision.ops import box_convert

parser = argparse.ArgumentParser(description="Few Shot Counting Demo code")
# parser.add_argument("-gdc", "--gd-config", type = str, help="Path to grounding dino config file")
# parser.add_argument("-ckp", "--gd-checkpoint", type = str, help="Path to grounding dino check point")
parser.add_argument("-i", "--input-image", type=str, required=True, help="/Path/to/input/image/file/")
parser.add_argument("-b", "--bbox-file", type=str, help="/Path/to/file/of/bounding/boxes")
parser.add_argument("-o", "--output-dir", type=str, default="./data/Result", help="/Path/to/output/image/file")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")

parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")

args = parser.parse_args()

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
regressor = CountRegressor(6, pool='mean')
checker = CountRegressor(6, pool='mean')
if use_gpu:
    resnet50_conv.cuda()
    regressor.cuda()
    regressor.load_state_dict(torch.load(args.model_path))
else:
    regressor.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

resnet50_conv.eval()
regressor.eval()

image_name = os.path.basename(args.input_image)
image_name = os.path.splitext(image_name)[0]
CONFIG_PATH = './groundingdino/config/GroundingDINO_SwinT_OGC.py'
CHECK_POINT_PATH = './weights/groundingdino_swint_ogc.pth'
# CONFIG_PATH = args.gd-config
# CHECK_POINT_PATH = args.gd-checkpoint
GD = load_model(CONFIG_PATH, CHECK_POINT_PATH)

def filter_large_boxes(boxes, max_size_ratio=0.1):
    filtered_boxes = []

    # Tính kích thước trung bình của bounding boxes
    average_size = sum([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]) / len(boxes)

    # Lọc các bounding box
    for box in boxes:
        box_size = (box[2] - box[0]) * (box[3] - box[1])
        if box_size <= max_size_ratio * average_size:
            filtered_boxes.append(box)

    return filtered_boxes


if args.bbox_file is None: # if no bounding box file is given, prompt the user for a set of bounding boxes
    # out_bbox_file = "{}/{}_box.txt".format(args.output_dir, image_name)
    # fout = open(out_bbox_file, "w")

    # im = cv2.imread(args.input_image)
    # cv2.imshow('image', im)
    # rects = select_exemplar_rois(im)
    IMAGE_PATH = args.input_image
    TEXT_PROMPT = input("Enter classes: ")
    BOX_TRESHOLD = 0.05
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, _, _ = predict(
        model=GD,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    rects = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    rects = filter_large_boxes(rects)
    # if len(rects)>5:
    #     rects = rects[1:4]
    rects1 = list()
    for rect in rects:
        x1, y1, x2, y2 = rect
        y1 = int(y1)
        x1 = int(x1)
        y2 = int(y2)
        x2 = int(x2)
        rects1.append([y1, x1, y2, x2])
        #fout.write("{} {} {} {}\n".format(y1, x1, y2, x2))

else:
    with open(args.bbox_file, "r") as fin:
        lines = fin.readlines()

    rects1 = list()
    for line in lines:
        data = line.split()
        y1 = int(data[0])
        x1 = int(data[1])
        y2 = int(data[2])
        x2 = int(data[3])
        rects1.append([y1, x1, y2, x2])

print("Bounding boxes: ", end="")
print(rects1)

image = Image.open(args.input_image)
image.load()
sample = {'image': image, 'lines_boxes': rects1}
sample = Transform(sample)
image, boxes = sample['image'], sample['boxes']


if use_gpu:
    image = image.cuda()
    boxes = boxes.cuda()

with torch.no_grad():
    features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

if not args.adapt:
    with torch.no_grad(): output = regressor(features)
else:
    features.required_grad = True
    #adapted_regressor = copy.deepcopy(regressor)
    adapted_regressor = regressor
    adapted_regressor.train()
    optimizer = optim.Adam(adapted_regressor.parameters(), lr=args.learning_rate)

    pbar = tqdm(range(args.gradient_steps))
    for step in pbar:
        optimizer.zero_grad()
        output = adapted_regressor(features)
        lCount = args.weight_mincount * MincountLoss(output, boxes, use_gpu=use_gpu)
        lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8, use_gpu=use_gpu)
        Loss = lCount + lPerturbation
        # loss can become zero in some cases, where loss is a 0 valued scalar and not a tensor
        # So Perform gradient descent only for non zero cases
        if torch.is_tensor(Loss):
            Loss.backward()
            optimizer.step()

        pbar.set_description('Adaptation step: {:<3}, loss: {}, predicted-count: {:6.1f}'.format(step, Loss.item(), output.sum().item()))

    features.required_grad = False
    output = adapted_regressor(features)


print('===> The predicted count is: {:6.2f}'.format(output.sum().item()))

rslt_file = "{}/{}_out.png".format(args.output_dir, image_name)
visualize_output_and_save(image.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file)
print("===> Visualized output is saved to {}".format(rslt_file))


