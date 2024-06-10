import numpy as np
import albumentations as A


class BoundingBoxCropper:
    ''' crop input image with provided bounding box
    '''
    def __call__(self, img, bbox):
        ''' bbox format as [ymin, xmin, ymax, xmax]
        '''

        height, width = img.shape[:2]
        bbox = [
            int(bbox[0] * height),
            int(bbox[1] * width),
            int(bbox[2] * height),
            int(bbox[3] * width)
        ]

        img = A.crop(img, x_min=bbox[1], y_min=bbox[0], x_max=bbox[3], y_max=bbox[2])
        return img