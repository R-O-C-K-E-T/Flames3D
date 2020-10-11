# Flames3D

A 3D Fractal Flame renderer that runs on the GPU.


To move around in the interactive tools the controls use WASD for XZ movement
and QE for Y movement. For rotating the camera use the number pad.

## Requirements
 - Python 3
 - FFmpeg (animation.py and randomised.py only)
 - An OpenGL 4.4 compatible graphics card (ideally with serveral GB of VRAM)

## Tools
 - fractalflames3d.py - Interactively render a flame
 - render.py - Renders flames to an image file
 - animation.py - Renders a flame animation
 - animation_ui.py - Interactively render a flame animation
 - randomised_animation.py - Renders a random flame animation
 - randomised.py - Interactively create random flames
 
 ## Todo
  - Add a better flame creation tool
  - Add workaround when glClearTexImage isn't present to lower the minimum OpenGL version to 4.2
  - Less sketchy usage of FFmpeg
