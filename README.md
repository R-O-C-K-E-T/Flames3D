# Flames3D

A 3D Fractal Flame renderer that runs on the GPU.

## Requirements
 - Python 3
 - FFmpeg (animation.py only)
 - An OpenGL 4.4 compatible graphics card (ideally with serveral GB of VRAM)

## Tools
 - fractalflames3d.py - Interactively render a flame
 - render.py - Renders flames to an image file
 - animation.py - Renders a flame animation
 - animation_ui.py - Interactively render a flame animation
 
 ## Todo
  - Add a better flame creation tool
  - Add workaround when glClearTexImage isn't present to lower the minimum OpenGL version to 4.2
  - Less sketchy usage of FFmpeg
