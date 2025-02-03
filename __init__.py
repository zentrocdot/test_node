from .nodes.nodes import *
from .node.nodes import DisplayData, ShowText
NODE_CLASS_MAPPINGS = { 
    "Circle Detection": CircleDetection,
    "Display Data": DisplayData,
    "Show Text": ShowText,
    }
    
print("\033[34mComfyUI Circle Detection Nodes: \033[92mLoaded\033[0m")
