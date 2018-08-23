from PIL import Image
import numpy as np

def load_image(path, shape=None, scale=1.0, args=None):
    img = Image.open(path)
    print("loading image: " + path)

    # if shape is not None and False:
    if shape is not None:
        # crop to obtain identical aspect ratio to shape
        width, height = img.size
        target_width, target_height = shape[0], shape[1]

        aspect_ratio = width / float(height)
        target_aspect = target_width / float(target_height)

        if aspect_ratio > target_aspect: # if wider than wanted, crop the width
            new_width = int(height * target_aspect)
            if args.crop == 'right':
                img = img.crop((width - new_width, 0, width, height))
            elif args.crop == 'left':
                img = img.crop((0, 0, new_width, height))
            else:
                img = img.crop(((width - new_width) / 2, 0, (width + new_width) / 2, height))
        else: # else crop the height
            new_height = int(width / target_aspect)
            if args.crop == 'top':
                img = img.crop((0, 0, width, new_height))
            elif args.crop == 'bottom':
                img = img.crop((0, height - new_height, width, height))
            else:
                img = img.crop((0, (height - new_height) / 2, width, (height + new_height) / 2))

        # resize to target now that we have the correct aspect ratio
        img = img.resize((target_width, target_height))
        print(target_height)
        print(target_width)
        img = img.resize((int(target_width), int(target_height)))

    # rescale
    w,h = img.size
    # img = img.resize((int(w * scale), int(h * scale)))
    # img.show()

    img_ = np.array(img)

    img_ = convertRGB2BGR(img_) / 255.0

    return img_

def convertRGB2BGR(img):
    # VGG_MEAN = [103.939, 116.779, 123.68]
    tmp = np.zeros(img.shape)
    tmp[:,:,0] = img[:,:,2]
    tmp[:,:,1] = img[:,:,1]
    tmp[:,:,2] = img[:,:,0]
    return tmp
