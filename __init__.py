from .nodes.nodes import *

NODE_CLASS_MAPPINGS = { 
    "Circle Detection": CircleDetection,
    "Display Any": RgthreeDisplayAny,
    "Display Int": RgthreeDisplayInt,
    }
    
print("\033[34mComfyUI Circle Detection Nodes: \033[92mLoaded\033[0m")
