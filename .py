import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
from jetbot import Robot, Camera, bgr8_to_jpeg
import traitlets
import ipywidgets.widgets as widgets

STEERING_GAIN = 0.2
SPEED = 0.4

# Class for handling model files
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

device = torch.device('cuda')

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

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

# Preprocess an image
def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(torch.device('cuda')).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# Display the camera image
def display_camera():
    camera = Camera.instance(width=224, height=224, fps=10)
    image_widget = ipywidgets.Image()
    traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
    display(image_widget)

robot = Robot()

# Compute steering value
def compute_steering_val(model, x):
    model = preprocess(x)
    xy = model(model)
    y = (0.5 - xy[1]) / 2.0
    return np.arctan2(xy[0], y)

# Calculate PID
def calc_pid(angle, angle_last):
    return angle * STEERING_GAIN + (angle - angle_last) * STEERING_GAIN

# Control motors
def control_motors(model, angle, angle_last):
    pid = calc_pid(compute_steering_val(model), angle_last)
    robot.left_motor.value = max(min(SPEED + pid, 1.0), 0.0)
    robot.right_motor.value = max(min(SPEED - pid, 1.0), 0.0)

# Normalize output vector
def normalize_output_vector(value):
    return F.softmax(value, dim=1)

# Check if blocked
def check_blocked(value):
    prob_blocked = float(value.flatten()[0])
    if prob_blocked < 0.5:
        return False
    else:
        return True

# Move forward
def move_forwards(value, inst_robot):
    if value:
        inst_robot.forward(speed)
    else:
        inst_robot.stop()

# Update based on camera change
def update(change, model, robot_inst, angle_last):
    x = change['new']
    x = preprocess(x)
    y = normalize_output_vector(model(x))
    blocked = checked_blocked(y)
    move_forwards(blocked, robot_inst)
    control_motors(model, compute_steering_val(model, x), angle_last)

# Initialize update
def init_update():
    update({'new': camera.value}, steering_model_weights_transfered, robot, 0.0)

# Move the robot
def move():
    camera.observe(init_update(), names='value')
