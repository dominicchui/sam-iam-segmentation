import os
import cv2 
import numpy as np
from tkinter import *
from PIL import ImageTk, Image, ImageSequence
from tkinter import filedialog

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img(predictor):
    global img
    global points
    points = []
    filename = openfn()
    img = cv2.imread(filename, 1)
  
    # displaying the image 
    cv2.imshow('image', img) 

    # Crop the image
    cv2.setMouseCallback('image', click_event, param = "crop")
  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cropped_image = img
    # cv2.imshow('cropped_image', cropped_image)

    # original image
    # original_image = convert_image_cv_to_tk(cropped_image)

    img_points = points
    img = segment_image(predictor, cropped_image, img_points)

    # image with selected points
    image = convert_image_cv_to_tk(img)
    panel = Label(root, image=image)
    panel.image = image
    panel.grid(row=1, column=0)

    # save cropped and segmented image


def segment_image(predictor, image, points):
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
    return make_mask(masks, image)

def make_mask(masks, image):
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

def open_vid():
    global img
    global points
    points = []
    filename = openfn()

    # convert gif to jpegs first
    output_dir = os.path.abspath(os.path.join(filename, "../frames"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gif_to_jpegs(filename, output_dir)

    # display frame 0 
    frame_0_path = os.path.abspath(os.path.join(output_dir, "0.jpeg"))
    img = cv2.imread(frame_0_path, 1) 
    cv2.imshow('image', img) 
  
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event, param = "crop")
  
    # wait for enter key to be pressed to exit 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = convert_image_cv_to_tk(img)
    panel = Label(root, image=image)
    panel.image = image
    panel.grid(row=1, column=1)
    vid_points = points
    print(vid_points)

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
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:
        if params == "crop":
            print("cropping")

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

            # Crop the image
            img = img[y1:y2, x1:x2]

            # Display the cropped image
            cv2.destroyAllWindows()
            cv2.imshow("image", img)

            # Save the cropped image
            output_dir = get_output_dir()
            cv2.imwrite(os.path.join(output_dir, "cropped_input_image.jpeg"), img)

            cv2.setMouseCallback('image', click_event, param = "segmentation")

        elif params == "segmentation":
            print("segmentation")
            points.append([x,y])
            print(f"({x}, {y})")

            # draw cross
            cv2.line(img,(x-5,y),(x+5,y),(255,255,255),2)
            cv2.line(img,(x,y-5),(x,y+5),(255,255,255),2)

            cv2.imshow('image', img)

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


# driver function 
if __name__=="__main__": 
    sam_enabled = False
    # # set up SAM
    if sam_enabled:
        sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        img_predictor = SAM2ImagePredictor(sam2_model)
    else:
        img_predictor = "test"

    root = Tk()
    root.geometry("1200x800+200+100")
    root.resizable(width=True, height=True)

    button_img = Button(root, text='Select Image', command= lambda: open_img(img_predictor))
    button_vid = Button(root, text="Select Video", command=open_vid)

    button_img.grid(row=0, column=0)
    button_vid.grid(row=0, column=1)

    root.mainloop()

