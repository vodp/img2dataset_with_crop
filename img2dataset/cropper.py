import numpy as np, json
import albumentations as A


class BoundingBoxCropper:
    ''' crop input image with provided bounding box
    '''
    def __call__(self, img, bbox):
        ''' bbox format as [ymin, xmin, ymax, xmax]
        '''
        try:
            bbox = json.loads(bbox)
        except:
            bbox = [0., 0., 1., 1.]
        height, width = img.shape[:2]
        
        xmin = int(bbox[1] * width)
        ymin = int(bbox[0] * height)
        xmax = int(bbox[3] * width)
        ymax = int(bbox[2] * height)

        img = A.crop(img, x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax)
        return img
