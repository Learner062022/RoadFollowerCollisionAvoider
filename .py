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
from typing import Union
import math

class ModelFile:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def load_trained_weights(self, model: torch.nn.Module):
        model.load_state_dict(torch.load(self._name))

INPUT_SIZE = 512
OUTPUT_SIZE = 2

def init_resnet18_model() -> torch.nn.Module:
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
    return model

def transfer_weights_to_model(file: ModelFile) -> torch.nn.Module:
    model = init_resnet18_model()
    file.load_trained_weights(model)
    device = torch.device(DEVICE)
    model = model.to(device)
    model = model.eval().half()
    return model

# Initialize model files and normalize parameters
follow_road = ModelFile('best_model_resnet18.pth')
avoid_collision = ModelFile('best_steering_model_xy.pth')

def create_rgb_tensor(red: float, green: float, blue: float) -> torch.Tensor:
    return torch.Tensor([red, green, blue]).cuda().half()

# Load and transfer models' weights
steering_model_weights_transfered = transfer_weights_to_model(follow_road)
collision_model_weights_transfered = transfer_weights_to_model(avoid_collision)

mean = create_rgb_tensor(0.485, 0.456, 0.406)
std = create_rgb_tensor(0.229, 0.224, 0.225)

# Preprocess an image
def preprocess(image: Union[np.ndarray, PIL.Image.Image]) -> torch.Tensor:
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(torch.device('cuda')).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

CAMERA_WIDTH = 224
CAMERA_HEIGHT = 224
FPS = 10

# Initialiase the camera
camera = Camera.instance(CAMERA_WIDTH, CAMERA_HEIGHT, FPS)
image_widget = widgets.Image()
traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
display(image_widget)

robot = Robot()

def create_slider(min_val: float, max_val: float, step_val: float = None, df: float = None, desc: str, orien: str = None) -> ipywidgets.FloatSlider: 
    if (default_val != None):
        return ipywidgets.FloatSlider(min=min_val, max=max_val, step=step_val, desc)
    elif (orien != None):
        return ipywidgets.FloatSlider(min=min_val, max=max_val, orientation=orien, desc)
    else:
        return ipywidgets.FloatSlider(min=min_val, max=max_val, step=step_val, value=df, desc)
    
speed_gain_slider = create_slider(0.0, 1.0, 0.01, 'speed gain')
steering_gain_slider = create_slider(0.0, 1.0, 0.01, 0.04, 'steering gain')
steering_dgain_slider = create_slider(0.0, 0.5, 0.001, 0.0, 'steering kd')
steering_bias_slider = create_slider(-0.3, 0.3, 0.01, 0.0, 'steering bias')

display(speed_control_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider)

blocked_slider = create_slider(0.0, 1.0, 'horizontal', 'blocked')
stopduration_slider= create_slider(1, 1000, 1, 10, 'time for stop') 
blocked_threshold= create_slider(0, 1.0, 0.01, 0.8, 'blocked threshold')

display(image_widget)

display(ipywidgets.HBox([blocked_slider, blocked_threshold, stopduration_slider]))

MAX_MOTOR_SPEED = 1.0
MIN_MOTOR_SPEED = 0.0

angle = 0.0
angle_last = 0.0
count_stops = 0
go_on = 1
stop_time = 10
x = 0.0
y = 0.0
speed_value = speed_control_slider.value

# Create function getting called whenever camera's value changes:
# 1. Pre-process the camera image
# 2. Execute the neural network models for Road following and Collision Avoidance
# 3. Check an if statements which would allow the jetbot to perform road following and stop whenever an obstacle has been detected
# 4. Compute the approximate steering value
# 5. Control the motors using proportional / derivative control (PD)
# Used in function:
# prob_blocked = float(F.softmax(collision_model_weights_transfered(x), dim=1).flatten()[0])

pid = angle * STEERING_GAIN + (angle - angle_last) * STEERING_GAIN
robot.right_motor.value = max(min(SPEED - pid, MAX_MOTOR_SPEED), MIN_MOTOR_SPEED)
robot.left_motor.value = max(min(SPEED + pid, MAX_MOTOR_SPEED), MIN_MOTOR_SPEED)

camera.observe(init_update(), names='value')
camera.unobserve(execute, names='value')
time.sleep(SLEEP_TIME)
robot.stop()