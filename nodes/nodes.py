#!/usr/bin/python

# Import the Python modules.
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
import json

# Tensor to PIL function.
def tensor2pil(image):
    '''Tensor to PIL image.''' 
    # Return PIL image.
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor function.
def pil2tensor(image):
    '''PIL image to tensor.'''
    # Return tensor.
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class AnyType(str):
    '''A special class that is always equal in not equal comparisons. Credit to Rgthree / pythongosssss'''

    def __ne__(self, __value: object) -> bool:
        return False

any = AnyType("*")

class CircleDetection:
    '''Circle detection node.'''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_canny_edge": ("FLOAT", {"default": 50, "min": 0, "max": 2048}),
                "threshold_circle_center": ("FLOAT", {"default": 30, "min": 0, "max": 2048}),
                "minR": ("INT", {"default": 1, "min": 0, "max": 2048}),
                "maxR": ("INT", {"default": 512, "min": 0, "max": 2048}),
                "dp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0}),
                "minDist": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 2048.0}),
                "color_tuple": ("STRING", {"multiline": False, "default": "(255, 0, 255)"}),
                "thickness": ("INT", {"default": 2, "min": 0, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING",)
    #RETURN_NAMES = ("IMAGE", "MASK, "TEXT",)
    FUNCTION = "circle_detection"
    CATEGORY = "🧬 Object Detection Nodes"
    OUTPUT_NODE = True
    
    def draw_circles(self, img, detected_circles, debug, color_tuple_str, thickness):
        '''Draw circles.'''
        outstr = ""
        print(color_tuple_str)
        # Copy image to a new image. 
        newImg = img.copy()
        strippedText = str(color_tuple_str).replace('(','').replace(')','').strip()
        print(strippedText)
        rgb = strippedText.split(",")
        print(rgb)
        r,g,b = int(rgb[0].strip()), int(rgb[1].strip()), int(rgb[2].strip()) 
        color_tuple = (r,g,b)  
        #COLOR_TUPLE = (255, 0, 255)
        #THICKNESS = 5
        # Declare local variables.
        a, b, r = 0, 0, 0
        # Draw detected circles.
        if detected_circles is not None:
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            print(detected_circles)
            # Loop over the detected circles.
            count = 0
            for pnt in detected_circles[0, :]:
                count += 1
                # Get the circle data.
                a, b, r = pnt[0], pnt[1], pnt[2]
                # Draw the circumference of the circle.
                cv2.circle(newImg, (a, b), r, color_tuple, thickness)
                # Draw a small circle of radius 1 to show the center.
                cv2.circle(newImg, (a, b), 1, color_tuple, 3)
                # Print dimensions and radius.
                if debug: 
                    print("No.:", count, "x:", a, "y", b, "r:", r)
                    outstr = outstr + "No. " + str(count) + " x: " + str(a) + " y: " + str(b) + " r: " + str(r) + "\n" 
        # Return image, co-ordinates and radius.
        return newImg, (a, b, r), outstr

    def pre_img(self, img):
        '''Preprocess image.'''
        # Set some file names.
        #img_1st = "gray_original.jpg"
        #img_2nd = "gray_blurred.jpg"
        # Convert image to grayscale.
        gray_org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Write image to file.
        #if debug:
        #cv2.imwrite(img_1st, gray_org)
        # Blur image using a 3x3 kernel.
        kernel = (3, 3)
        gray_blur = cv2.blur(gray_org, kernel)
        # Write image to file.
        #if debug:
        #cv2.imwrite(img_2nd, gray_blur)
        # Return blurred gray image.
        return gray_blur

    def detect_circles(self, gray_blur, threshold_canny_edge, threshold_circle_center, minR, maxR, minDist, dp, debug):
        '''Detect circles.'''
        # Apply a Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blur,
                       cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                       param1=threshold_canny_edge,
                       param2=threshold_circle_center,
                       minRadius=minR, maxRadius=maxR)
        # Print detected data.
        if debug:
            print("Detected circles:", detected_circles)
        # Return detected_circles.
        return detected_circles

    def post_img(self, img, detected_circles, debug, color_tuple, thickness):
        '''Postprocess image.'''
        # Draw circles.
        img, (a, b, r), outstr = self.draw_circles(img, detected_circles, debug, color_tuple, thickness)
        # Return image and tuple.
        return img, (a, b, r), outstr

    def circle_detection(self, image, threshold_canny_edge, threshold_circle_center, minR, maxR, minDist, dp, color_tuple, thickness):
        '''Main script function.'''
        # Print detection parameters.
        print("Threshold canny edge:", threshold_canny_edge)
        print("Threshold circle center:", threshold_circle_center)
        print("minR:", minR)
        print("maxR:", maxR)
        print("minDist:", minDist)
        print("dp:", dp)
        # Set the debug flag.
        debug = True
        # Create PIL image.
        img_input = tensor2pil(image)
        # Create numpy array.
        img_input = np.asarray(img_input)
        # Preprocess image.
        gray_blur = self.pre_img(img_input)
        # Process image. Detect circles.
        detected_circles = self.detect_circles(gray_blur, threshold_canny_edge, threshold_circle_center, minR, maxR, minDist, dp, debug)
        # Postrocess image.
        img_output, _, out_string = self.post_img(img_input, detected_circles, debug, color_tuple, thickness)
        # Create output image.
        img_output = Image.fromarray(img_output)
        # Create tensor.
        image_out = pil2tensor(img_output)
        # Create simple mask for testing purposes.
        out_mask = torch.zeros((64,64), dtype=torch.float32, device="cpu") 
        # Return None.
        return (image_out, out_mask, out_string,)

class DisplayData:
    '''Display any data node.'''

    # NAME = "Display"
    CATEGORY = "🧬 Object Detection Nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (any, {}),
            },
        }

    RETURN_TYPES = ()
    RETURN_TYPES = ("STRING",)
    FUNCTION = "display_data"
    OUTPUT_NODE = True
    

    def display_data(self, source=None):
        '''Display data.'''
        print("Source: Gotcha!")
        value = 'None'
        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)):
            value = str(source)
        elif source is not None:
            try:
                value = json.dumps(source)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = 'Source exists, but could be displayed.'
        print("Source:", source) 
        value = "test"
        return {"ui": {"text": (value,)}}
