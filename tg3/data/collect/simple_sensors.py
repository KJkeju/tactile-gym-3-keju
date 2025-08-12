import cv2
import ipdb
from tg3.data.process.transform_image import transform_image


class SimSensor:
    def __init__(self, sensor_params={}, embodiment={}):
        self.sensor_params = sensor_params
        self.embodiment = embodiment

    def read(self):
        img = self.embodiment.get_tactile_observation()
        return img

    def process(self, outfile=None):
        img = self.read()

        img_left = img['left']
        img_right = img['right']
        img_left = transform_image(img_left, **self.sensor_params)
        img_right = transform_image(img_right, **self.sensor_params)

        if outfile:
            cv2.imwrite(outfile, img)
        return img_left, img_right


class RealSensor:
    def __init__(self, sensor_params={}):
        self.sensor_params = sensor_params
        source = sensor_params.get('source', 0)
        exposure = sensor_params.get('exposure', -7)

        self.cam = cv2.VideoCapture(source)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(5):
            self.cam.read()  # Hack - initial camera transient

    def read(self):
        # self.cam.read()  # Hack - throw one away - buffering issue (note - halves frame rate!)
        _, img = self.cam.read()
        return img

    def process(self, outfile=None):
        img = self.read()
        img = transform_image(img, **self.sensor_params)
        if outfile:
            cv2.imwrite(outfile, img)
        return img


class ReplaySensor:
    def __init__(self, sensor_params={}):
        self.sensor_params = sensor_params

    def read(self, outfile):
        img = cv2.imread(outfile)
        return img

    def process(self, outfile):
        img = self.read(outfile)
        # img = transform_image(img, **self.sensor_params)
        return img


class DummySensor:
    def __init__(self, sensor_params={}):
        self.sensor_params = sensor_params

    def read(self, outfile):
        img = None
        return img

    def process(self, outfile):
        img = None
        return img
    