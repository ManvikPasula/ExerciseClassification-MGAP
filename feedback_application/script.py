import cv2
import time
import tkinter as tk
import torch
import torch.nn as nn
from torch.nn import Linear
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from torch.nn.functional import softmax
import torch.utils.data
import os
import pytorch_lightning

model1_name = "slowfast_r50"
model2_name = "x3d_m"

os.chdir('/Users/TMP/Desktop/test2')

model1_best_path = 'checkpoints/slowfast.ckpt'
model2_best_path = 'checkpoints/x3d.ckpt'
num_classes = 16
num_labels = 11
batch_size = 8
num_workers = 4
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
clip_duration = (num_frames * sampling_rate) / frames_per_second
# device=('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
pretrained = True
learning_rate = 0.0001
dropout_rate = 0.6
gamma = 2

pred_to_class = {
    0: "triceps",
    1: "lats",
    2: "biceps",
    3: "quads",
    4: "glutes",
    5: "shoulders",
    6: "abs",
    7: "obliques",
    8: "chest",
    9: "lower back",
    10: "hamstrings",
}

id_to_exercise = {
    0: "bench press",
    1: "bicep curl",
    2: "chest fly machine",
    3: "deadlift",
    4: "hip thrust",
    5: "lat pulling",
    6: "lateral raise",
    7: "leg extension",
    8: "leg raises",
    9: "push-up",
    10: "russian twist",
    11: "shoulder press",
    12: "squat",
    13: "t bar row",
    14: "tricep Pushdown",
    15: "tricep dips",
}

class_to_label = {
    0: [8, 5, 0],
    1: [2],
    2: [8],
    3: [4, 9, 10],
    4: [4],
    5: [1, 2],
    6: [5],
    7: [3],
    8: [6],
    9: [8, 5, 0],
    10: [6, 7],
    11: [5],
    12: [3, 4, 10],
    13: [1, 2],
    14: [0],
    15: [0],
}

id_to_label = {
    0: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    1: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    3: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    4: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    5: [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    6: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    7: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    9: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    10: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    11: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    12: [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    13: [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    14: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    15: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


slowfast_transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)


def post_act(input):
    return softmax(input, dim=1)


class x3d_VideoClassificationLightningModule(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = torch.hub.load('facebookresearch/pytorchvideo', model2_name, pretrained=True)
        self.model.to(device)
        self.model.blocks[5].proj = Linear(in_features=2048, out_features=num_labels, bias=True)
        self.model.train()

    def forward(self, x):
        return self.model(x)


class slowfast_VideoClassificationLightningModule(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/pytorchvideo", model=model1_name, pretrained=True)
        self.model.to(device)
        self.model.blocks[6].proj = nn.Linear(in_features=2304, out_features=num_labels, bias=True)
        self.model.train()

    def forward(self, x):
        return self.model(x)


torch.set_float32_matmul_precision('medium')

model1 = slowfast_VideoClassificationLightningModule.load_from_checkpoint(model1_best_path)
model1.to(device)
model1.eval()

model2 = x3d_VideoClassificationLightningModule.load_from_checkpoint(model2_best_path)
model2.model.blocks[5].activation = None
model2.to(device)
model2.eval()

def get_results():
    video_path = "videos/output.mp4"
    os.chdir('/Users/TMP/Desktop/test2')

    start_sec = 0
    end_sec = start_sec + 2

    video = EncodedVideo.from_path(video_path)
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    video_data = slowfast_transform(video_data)

    # this is here to wipe the videos folder after every use
    #os.remove(video_path)

    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]

    slowfast_data = inputs
    x3d_data = inputs[1]

    y_hat1 = model1(slowfast_data)
    y_hat2 = model2(x3d_data)

    y_hat = []
    for x in range(0, y_hat1.shape[0]):
        temp = []
        for y in range(0, 11):
            val = 0.75 * y_hat1[x][y] + 0.25 * y_hat2[x][y]
            temp.append(val)
        y_hat.append(temp)

    preds = torch.Tensor(y_hat)
    preds = post_act(preds)

    # k = int(input("How many of the top classes do you want to see? (max is 11) : ")) # k can be at max 11
    k = 11

    pred_classes = preds.topk(k=k).indices
    pred_class_names = [pred_to_class[int(i)] for i in pred_classes[0]]

    pred_values = preds.topk(k=k).values[0]

    # with open("activation predictions.txt", "a") as f:
    results = "Here are your top 3 most activated muscle groups:\n"
    pred_with_softmax = {}
    for x in range(0, 3):
        prob_val = round(pred_values[x].item() * 100, 2)
        pred_with_softmax[pred_class_names[x]] = prob_val

        if prob_val < 0.2:
            break

        results += str(prob_val) + "% chance for " + pred_class_names[x] + " activation.\n"
    results += "\nCompare to the ideal activations for this exercise to understand what to target for proper form."
    # return results
    return results


def resultsAssigner():
    global results_label
    '''
    results_label.pack_forget()
    results_label = tk.Label(root, text='model is running', font=("Helvetica", 24))
    results_label.pack(expand=True)
    resultsAnswer = get_results()
    print(resultsAnswer)
    results_label.pack_forget()
    results_label = tk.Label(root, text=resultsAnswer, font=("Helvetica", 24))
    results_label.pack(expand=True)
    '''
    resultsAnswer = get_results()
    results_label.config(text=resultsAnswer)
    root.update()


def start_recording():
    results_label.config(text='No results yet')
    root.update()
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('videos/output.mp4', fourcc, 20.0, (screen_width, screen_height))
    # Countdown before starting recording
    for i in range(3, 0, -1):
        countdown_label.config(text=str(i))
        root.update()
        time.sleep(1)

    countdown_label.config(text="0")
    root.update()

    # Start recording
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize the frame to fit the screen
            frame = cv2.resize(frame, (screen_width, screen_height))

            # Write the frame
            out.write(frame)
            cv2.imshow('frame', frame)

            # Check if 3 seconds have elapsed
            if time.time() - start_time > 3:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Hit q to exit
                break
        else:
            break

    # Release everything if the job is finished
    out.release()
    cap.release()
    cv2.destroyAllWindows()

# Create a Tkinter window
root = tk.Tk()
root.title("Muscle Activation Predictor")

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to full screen
root.attributes('-fullscreen', True)

# Create a label to display the countdown
title = tk.Label(root, text="Manvik Pasula, Muscle Group Activation Predictor", font=("Helvetica", 36))

countdown_label = tk.Label(root, font=("Helvetica", 40))

# Create a label for results with placeholder text

results_label = tk.Label(root, text='No results yet', font=("Helvetica", 24))
start_button = tk.Button(root, text="Start Recording", font=("Helvetica", 24), bg='#A9A9A9', command=start_recording)

results_button = tk.Button(root, text="Get Results", font=("Helvetica", 24), bg='#A9A9A9', command=resultsAssigner)
results_text = tk.Label(root, text="RESULTS:", font=("Arial", 28))

title.pack(expand=True)
start_button.pack(expand=True)
countdown_label.pack(expand=True)
results_button.pack(expand=True)
results_text.pack(expand=True)
results_label.pack(expand=True)


# Get the width and height of the frame
#cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('videos/output.mp4', fourcc, 20.0, (screen_width, screen_height))


# Button to start recording
while True:
    # start_button = tk.Button(root, text="Start Recording", font=("Helvetica", 24), bg='#A9A9A9', command=start_recording)
    # start_button.pack(expand=True)
    #
    # results_button = tk.Button(root, text="Get Results", font=("Helvetica", 24), bg='#A9A9A9', command=resultsAssigner)
    # results_button.pack(expand=True)
    #
    # #Label for countdown
    # countdown_label.pack(expand=True)
    #
    # #Label for results
    # results_text.pack(expand=True)
    # results_label.pack(expand=True)

    root.mainloop()
