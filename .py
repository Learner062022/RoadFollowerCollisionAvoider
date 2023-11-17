import torch
import time
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from IPython.display import display
import cv2
import PIL.Image
import numpy as np
from jetbot import Robot, Camera, bgr8_to_jpeg
import traitlets
import ipywidgets.widgets as widgets

STEERING_GAIN = 0.2
SPEED = 0.4

# Handles model files
class ModelFile:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def load_trained_weights(self, model):
        model.load_state_dict(torch.load(self._name))

# Initialize a ResNet18 model
def init_resnet18_model():
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    return model

# Transfer weights to a model
def transfer_weights(file: ModelFile):
    model = init_resnet18_model()
    file.load_trained_weights(model)
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval().half()
    return model

# Initialize model files and normalize parameters
follow_road = ModelFile('best_model_resnet18.pth')
avoid_collision = ModelFile('best_steering_model_xy.pth')

# Load and transfer models' weights
steering_model_weights_transfered = transfer_weights(follow_road)
collision_model_weights_transfered = transfer_weights(avoid_collision)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

# Preprocess an image
def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(torch.device('cuda')).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# Initialiase the camera
camera = Camera.instance(width=224, height=224, fps=10)
image_widget = widgets.Image()
traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
display(image_widget)

robot = Robot()

# Compute steering value
def compute_steering_val(model, x):
    xy = model(x).detach().float().cpu().numpy().flatten()
    return np.arctan2(xy[0], (0.5 - xy[1]) / 2.0)

# Calculate PID
def calc_pid(angle, angle_last):
    return angle * STEERING_GAIN + (angle - angle_last) * STEERING_GAIN

# Control motors
def control_motors(model, angle_last):
    x = camera.value
    x = preprocess(x)
    
    # Collision avoidance model
    prob_blocked = float(F.softmax(collision_model_weights_transfered(x), dim=1).flatten()[0])
    if prob_blocked > 0.5:
        robot.stop()
    
    # Road following model
    steering_val = compute_steering_val(model, x)
    pid = calc_pid(steering_val, angle_last)
    robot.left_motor.value = max(min(SPEED + pid, 1.0), 0.0)
    robot.right_motor.value = max(min(SPEED - pid, 1.0), 0.0)

# Update based on camera change
def update(change):
    control_motors(steering_model_weights_transfered, 0.0)

# Initialize update
def init_update():
    update({'new': camera.value})
    
time.sleep(0.1)

# Move the robot
def move():
    camera.observe(init_update(), names='value')
    
# Start moving the robot
move()
