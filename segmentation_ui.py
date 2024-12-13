import os
import shutil
from tkinter import messagebox
import cv2 
import numpy as np
from tkinter import *
from PIL import ImageTk, Image, ImageSequence
from tkinter import filedialog
import tomli
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import torch

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

# called when opening an image in tkinter
def open_img(predictor):
    global img
    global points
    points = []
    filename = openfn()
    img = cv2.imread(filename, 1)
    process_image(predictor, False)

# processes the selected img (global) and displays using tkinter
def process_image(predictor, from_video):
    global img
    # displaying the image 
    cv2.imshow('image', img) 

    # Crop the image
    if from_video:
        param = ["crop", "video"]
    else:
        param = ["crop", "image"]
    cv2.setMouseCallback('image', click_event, param)
  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cropped_image = img

    img_points = points
    img = segment_image(predictor, cropped_image, img_points, from_video)

    # image with selected points
    image = convert_image_cv_to_tk(img)
    panel = Label(root, image=image)
    panel.image = image
    if from_video:
        panel.grid(row=1, column=1)
    else:
        panel.grid(row=1, column=0)

# segments the image with the key points and returns the segmentation mask
# if SAM is not available, it returns a dummy mask
def segment_image(predictor, image, points, from_video):
    if sam_enabled:
        predictor.set_image(image)
        points = np.array(points)
        input_point = np.array(points)
        input_label = np.array([1] * points.shape[0])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
    else:
        # test mask
        width = image.shape[0]
        height = image.shape[1]
        mask = np.full((1, width, height), 0)
        mask[0, 50:100, 50:100] = 1
        masks = np.array(mask)
    return make_mask(masks, image, from_video)

# combines the image and mask into one image
# also saves the mask by itself to disk
def make_mask(masks, image, from_video):
    # convert mask into image
    mask = masks[0]
    # print("mask shape: ", mask.shape)
    color = np.array([30, 144, 255, 153])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2_mask = mask_image.astype(np.uint8)

    # save mask
    output_dir = get_output_dir()
    if from_video:
        cv2.imwrite(os.path.join(output_dir, "input_video_frame_0_mask.jpeg"), mask*255)
    else:
        cv2.imwrite(os.path.join(output_dir, "input_image_mask.jpeg"), mask*255)

    # compose image and color mask using alpha channel of mask
    cv2_img = image
    bg = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGBA))
    fg = Image.fromarray(cv2.cvtColor(cv2_mask, cv2.COLOR_BGRA2RGBA))
    x, y = ((bg.width - fg.width) // 2 , (bg.height - fg.height) // 2)
    bg.paste(fg, (x, y), fg)
    bg = bg.convert("RGB")

    cv2_merged = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
    return cv2_merged

# opens a video selected in tkinter
# currently only support .gif
def open_vid(predictor):
    global img
    global points
    points = []
    filename = openfn()

    # todo handle other video formats
    # convert gif to jpegs first
    output_dir = os.path.abspath(os.path.join(get_output_dir(), "original_video_frames"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gif_to_jpegs(filename, output_dir)

    # display frame 0 
    frame_0_path = os.path.abspath(os.path.join(output_dir, "0.jpeg"))
    img = cv2.imread(frame_0_path, 1) 

    process_image(predictor, True)
    vid_points = points
    print(vid_points)
    button_process_vid["state"] = NORMAL

# opens a directory of jpg frames as a video
def open_vid_frames(predictor):
    global img
    global points
    points = []

    source_dir = filedialog.askdirectory()
    print(f"dir_name {source_dir}")

    # copy over video frames
    output_dir = os.path.abspath(os.path.join(get_output_dir(), "original_video_frames"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(output_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_file)

    # display frame 0 
    frame_0_path = os.path.abspath(os.path.join(source_dir, "00000.jpg"))
    img = cv2.imread(frame_0_path, 1) 

    process_image(predictor, True)
    vid_points = points
    print(vid_points)
    button_process_vid["state"] = NORMAL

# convert cv2 image to tk acceptable format
def convert_image_cv_to_tk(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((500, 500), Image.Resampling.LANCZOS)
    image = ImageTk.PhotoImage(image)
    return image

def gif_to_jpegs(gif_path, output_dir):
    """Converts a GIF to a series of JPEGs."""

    with Image.open(gif_path) as im:
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            frame = frame.convert("RGB") 
            frame.save(f"{output_dir}/{i}.jpeg")
   
# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
    global img
    global points
    global crop
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:
        if "crop" in params:
            print("cropping")

            # Crop the image
            (x1, y1, x2, y2) = get_crop_coordinates(img, x, y)
            # save crop for video cropping
            if "video" in params:
                crop = (x1, y1, x2, y2)
            img = img[y1:y2, x1:x2]

            # Display the cropped image
            cv2.destroyAllWindows()
            cv2.imshow("image", img)

            # Save the cropped image
            output_dir = get_output_dir()
            if "image" in params:
                cv2.imwrite(os.path.join(output_dir, "cropped_input_image.jpeg"), img)
            else:
                cv2.imwrite(os.path.join(output_dir, "cropped_video_frame_0.jpeg"), img)

            cv2.setMouseCallback('image', click_event, param = "segmentation")

        elif "segmentation" in params:
            print("segmentation")
            points.append([x,y])
            print(f"({x}, {y})")

            # draw cross
            cv2.line(img,(x-5,y),(x+5,y),(255,255,255),2)
            cv2.line(img,(x,y-5),(x,y+5),(255,255,255),2)

            cv2.imshow('image', img)

def get_crop_coordinates(img, x, y):
    height, width = img.shape[:2]

    # Determine the smaller dimension to create a square crop
    crop_size = min(width, height) // 2
    
    # Calculate the crop coordinates
    x1 = x - crop_size
    x2 = x + crop_size
    y1 = y - crop_size
    y2 = y + crop_size
    # ensure crop is in bounds
    if x1 < 0:
        x2 -= x1
        x1 -= x1
    if y1 < 0:
        y2 -= y1
        y1 -= y1
    if x2 > width:
        diff = x2 - width
        x1 -= diff
        x2 -= diff
    if y2 > height:
        diff = y2 - height
        y1 -= diff
        y2 -= diff
    return (x1, y1, x2, y2)

def process_video(predictor, points):
    global crop
    x1, y1, x2, y2 = crop

    if sam_enabled:
        # SAM2 Segmentation
        video_dir = os.path.abspath(os.path.join(get_output_dir(), "original_video_frames"))
        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

        prompts = {}  # hold all the clicks we add for visualization

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # adding points
        points = np.array(points, dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1] * points.shape[0])
        prompts[ann_obj_id] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # Propagate segments
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # save the results
        result_dir = os.path.abspath(os.path.join(get_output_dir(), "segmentation_output"))
        print(result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for i in range(len(video_segments)):
            frame = np.squeeze(video_segments[i][1])
            # crop image before saving
            cropped_image = frame[y1:y2, x1:x2]
            im = Image.fromarray(cropped_image)
            im.save(os.path.join(result_dir, str(i)+".jpeg"))

    else:
        # if segmentation through SAM is not available, use canny edge instead
        # crop image first, then find canny edges
        output_dir = os.path.abspath(os.path.join(get_output_dir(), "cropped_video_frames"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        original_frames_dir = os.path.abspath(os.path.join(get_output_dir(), "original_video_frames"))
        for filename in os.listdir(original_frames_dir):
            filepath = os.path.join(original_frames_dir, filename)
            image = cv2.imread(filepath)

            # Crop the image
            cropped_image = image[y1:y2, x1:x2]

            edges = cv2.Canny(cropped_image, 100, 200)

            # Save cropped image
            cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jpeg"), edges)

    if tkinter_enabled:
        messagebox.showinfo("SAM-IAM", "Video Processed!")

def get_output_dir():
    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)

    # Get the directory of the script
    script_dir = os.path.dirname(script_path)

    output_dir = os.path.abspath(os.path.join(script_dir, "output"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_config_file_path():
    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)

    # Get the directory of the script
    script_dir = os.path.dirname(script_path)
    return os.path.join(script_dir, "config.toml")


# driver function 
if __name__=="__main__": 
    sam_enabled = True
    tkinter_enabled = False

    crop = (0, 0, 0, 0)

    # set up SAM
    if sam_enabled:

        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        img_predictor = SAM2ImagePredictor(sam2_model)
        vid_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    else:
        img_predictor = "test"
        vid_predictor = "test"

    if tkinter_enabled:
        root = Tk()
        root.title('SAM-IAM')
        root.geometry("1200x800+200+100")
        root.resizable(width=True, height=True)

        button_img = Button(root, text='Select Image', command=lambda: open_img(img_predictor))
        button_vid = Button(root, text="Select Video", command=lambda: open_vid(img_predictor))
        button_vid_frames = Button(root, text="Select Video Frames", command=lambda: open_vid_frames(img_predictor))
        button_process_vid = Button(root, text="Process Video", command=lambda: process_video(vid_predictor))
        button_process_vid["state"] = DISABLED

        button_img.grid(row=0, column=0)
        button_vid.grid(row=0, column=1)
        button_vid_frames.grid(row=0, column=2)
        button_process_vid.grid(row=2, column=0)

        root.mainloop()
    else:
        # Process based on config file

        # Read config file
        file_path = get_config_file_path()
        if not os.path.exists(file_path):
            print("config.toml file not found")
            exit
        with open(file_path, "rb") as f:
            data = tomli.load(f)

        # IMAGE
        if not os.path.exists(data["image"]["input_path"]):
            print("input image not found")
            exit
        image = cv2.imread(data["image"]["input_path"])

        # Crop:
        # the center point for the square cropping
        crop_center = data["image"]["crop_center"]
        (x1, y1, x2, y2) = get_crop_coordinates(image, crop_center[0], crop_center[1])
        cropped_image = image[y1:y2, x1:x2]

        output_dir = get_output_dir()
        cv2.imwrite(os.path.join(output_dir, "cropped_input_image.jpeg"), cropped_image)

        # Segment
        # the points for segmentation
        seg_points = data["image"]["points"]
        preview_img = segment_image(img_predictor, cropped_image, seg_points, False)

        # VIDEO
        video_path = data["video"]["input_path"]
        if not os.path.exists(video_path):
            print("input video not found")
            exit
    
        output_dir = os.path.abspath(os.path.join(get_output_dir(), "original_video_frames"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # check if path is to a file or dir
        if os.path.isdir(video_path):
            # copy frames over
            for filename in os.listdir(video_path):
                source_file = os.path.join(video_path, filename)
                destination_file = os.path.join(output_dir, filename)
                if os.path.isfile(source_file):
                    shutil.copy(source_file, destination_file)
            
            frame_0_path = os.path.abspath(os.path.join(output_dir, "00000.jpg"))
        else:
            # turn gif into frames
            gif_to_jpegs(video_path, output_dir)

            frame_0_path = os.path.abspath(os.path.join(output_dir, "0.jpeg"))

        # crop and segment frame 0 
        frame_0 = cv2.imread(frame_0_path)

        crop_center = data["video"]["crop_center"]
        (x1, y1, x2, y2) = get_crop_coordinates(frame_0, crop_center[0], crop_center[1])
        cropped_frame_0 = frame_0[y1:y2, x1:x2]

        seg_points = data["video"]["points"]
        preview_img = segment_image(img_predictor, cropped_frame_0, seg_points, True)
        crop = (x1, y1, x2, y2)
        process_video(vid_predictor, seg_points)
