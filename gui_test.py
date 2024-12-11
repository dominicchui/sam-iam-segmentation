import os
from tkinter import messagebox
import cv2 
import numpy as np
from tkinter import *
from PIL import ImageTk, Image, ImageSequence
from tkinter import filedialog
import tomli

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img(predictor):
    global img
    global points
    points = []
    filename = openfn()
    img = cv2.imread(filename, 1)
    process_image(predictor, False)

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


def segment_image(predictor, image, points, from_video):
    if sam_enabled:
        predictor.set_image(image)
        input_point = np.array(points)
        input_label = np.array([1])

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
    # print("masks shape: ", masks.shape)
    return make_mask(masks, image, from_video)

def make_mask(masks, image, from_video):
    # convert mask into image
    mask = masks[0]
    # print("mask shape: ", mask.shape)
    color = np.array([30, 144, 255, 153])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2_mask = mask_image.astype(np.uint8)
    # cv2_mask = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2BGRA)
    # cv2.imshow("cv2_mask", cv2_mask)

    # save mask
    output_dir = get_output_dir()
    if from_video:
        cv2.imwrite(os.path.join(output_dir, "input_video_frame_0_mask.jpeg"), mask*255)
    else:
        cv2.imwrite(os.path.join(output_dir, "input_image_mask.jpeg"), mask*255)

    # convert base image to cv2 image
    cv2_img = image
    # cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("cv2_img", cv2_img)

    # compose image and color mask using alpha channel of mask
    bg = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGBA))
    fg = Image.fromarray(cv2.cvtColor(cv2_mask, cv2.COLOR_BGRA2RGBA))
    x, y = ((bg.width - fg.width) // 2 , (bg.height - fg.height) // 2)
    bg.paste(fg, (x, y), fg)
    bg = bg.convert("RGB")

    cv2_merged = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
    # cv2.imshow("cv2_merged", cv2_merged)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cv2_merged

def open_vid(predictor):
    global img
    global points
    points = []
    filename = openfn()

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

def process_video():
    global crop
    # crop each frame
    x1, y1, x2, y2 = crop

    output_dir = os.path.abspath(os.path.join(get_output_dir(), "cropped_video_frames"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # todo actually want to crop masks instead
    original_frames_dir = os.path.abspath(os.path.join(get_output_dir(), "original_video_frames"))
    for filename in os.listdir(original_frames_dir):
        filepath = os.path.join(original_frames_dir, filename)
        image = cv2.imread(filepath)

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        # Save cropped image
        cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jpeg"), cropped_image)
    if tkinter_enabled:
        messagebox.showinfo("SAM-IAM", "Video Processed!")

def get_output_dir():
    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)

    # Get the directory of the script
    script_dir = os.path.dirname(script_path)

    output_dir = os.path.abspath(os.path.join(script_dir, "output"))
    # print("output_dir: ", output_dir)
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
    sam_enabled = False
    tkinter_enabled = False

    crop = (0, 0, 0, 0)
    # # set up SAM
    if sam_enabled:
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        img_predictor = SAM2ImagePredictor(sam2_model)
    else:
        img_predictor = "test"

    if tkinter_enabled:
        root = Tk()
        root.title('SAM-IAM')
        root.geometry("1200x800+200+100")
        root.resizable(width=True, height=True)

        button_img = Button(root, text='Select Image', command=lambda: open_img(img_predictor))
        button_vid = Button(root, text="Select Video", command=lambda: open_vid(img_predictor))
        button_process_vid = Button(root, text="Process Video", command=process_video)
        button_process_vid["state"] = DISABLED

        button_img.grid(row=0, column=0)
        button_vid.grid(row=0, column=1)
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
        if not os.path.exists(data["video"]["input_path"]):
            print("input video not found")
            exit

        output_dir = os.path.abspath(os.path.join(get_output_dir(), "original_video_frames"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        gif_to_jpegs(data["video"]["input_path"], output_dir)

        # crop and segment frame 0 
        frame_0_path = os.path.abspath(os.path.join(output_dir, "0.jpeg"))
        frame_0 = cv2.imread(frame_0_path)

        crop_center = data["video"]["crop_center"]
        (x1, y1, x2, y2) = get_crop_coordinates(frame_0, crop_center[0], crop_center[1])
        cropped_frame_0 = frame_0[y1:y2, x1:x2]

        seg_points = data["video"]["points"]
        preview_img = segment_image(img_predictor, cropped_frame_0, seg_points, True)
        crop = (x1, y1, x2, y2)
        process_video()
