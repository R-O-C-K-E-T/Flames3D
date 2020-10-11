import argparse, json, copy, os, math, os.path, shutil, time

import numpy as np

from fractalflames3d import rotationY_mat, rotationX_mat, rotationZ_mat

def de_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # assume no object arrays

    if isinstance(obj, list):
        return [de_numpy(item) for item in obj]

    if isinstance(obj, dict):
        return dict((de_numpy(key), de_numpy(val)) for key, val in obj.items())

    return obj

def zero_pad(digits, n):
    v = str(n)
    while len(v) < digits:
        v = '0' + v
    return v

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a 3d fractal flame animation')

    parser.add_argument('flame', type=argparse.FileType('r'), help='Base flame to animate')
    parser.add_argument('animation', type=argparse.FileType('r'), help='Python script to permute the flame')

    parser.add_argument('--fast', '-f', action='store_true', help='Flag to switch to low quality mode')
    parser.add_argument('--duration', '-d', type=float, default=20, help='Duration of output video')

    args = parser.parse_args()

    initial = time.time()

    tmp_folder = 'tmp'
    dest_folder = 'output'

    shutil.rmtree(tmp_folder)
    shutil.rmtree(dest_folder)

    os.makedirs(tmp_folder)
    os.makedirs(dest_folder)

    if args.fast:
        cycles = 500
        width = 500
        height = 500
        depth = 64
        framerate = 15
    else:
        cycles = 2000
        width = 1000
        height = 1000
        depth = 256
        framerate = 30

    duration = args.duration
    if duration <= 0:
        raise RuntimeError('Duration <= 0')

    frames = max(round(framerate * duration), 1)
    digits = math.ceil(math.log(frames, 10))

    original_flame = json.load(args.flame)
    code = compile(args.animation.read(), '<string>', 'exec')

    for i in range(frames):
        flame = copy.deepcopy(original_flame)
        t = i / frames

        for func in flame['functions']:
            func['pre_trans'] = np.array(func['pre_trans'])
            func['post_trans'] = np.array(func['post_trans'])
        
        exec(code, globals(), {})
        
        for func in flame['functions']:
            func['pre_trans'] = func['pre_trans'].tolist()
            func['post_trans'] = func['post_trans'].tolist()

        with open(os.path.join(tmp_folder, zero_pad(digits, i) + '.json'), 'w') as f:
            json.dump(flame, f)

    os.system('./render.py --dest {} --cycles {} --width {} --height {} --depth {} {}/*.json'.format(dest_folder, cycles, width, height, depth, tmp_folder))
    try:
        os.remove('out.mp4')
    except FileNotFoundError:
        pass
    os.system('ffmpeg -framerate {} -pattern_type glob -i \'{}/*.png\' -c:v libx264 -pix_fmt yuv420p out.mp4'.format(framerate, dest_folder))


    dt = time.time() - initial
    print('Animation done in {:.3f}s {:.3f}x'.format(dt, duration / dt))
