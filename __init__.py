from .nodes.nodes import *
from .nodes.nodes import DisplayData, ShowText

NODE_CLASS_MAPPINGS = { 
    "Circle Detection": CircleDetection,
    "Display Data": DisplayData,
    "Show Text": ShowText,
    "Print Hello World": PrintHelloWorld,
    }
    
print("\033[34mComfyUI Circle Detection Nodes: \033[92mLoaded\033[0m")
