from collections import deque

import numpy as np


DEBUG = False


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        if DEBUG:
            import cv2

            canvas = np.ones((100, 100, 3), dtype=np.uint8)
            w = int(canvas.shape[1] / len(self._window))
            h = 99

            for i in range(1, len(self._window)):
                y1 = (self._max - self._window[i-1]) / (self._max - self._min + 1e-8)
                y2 = (self._max - self._window[i]) / (self._max - self._min + 1e-8)

                cv2.line(
                        canvas,
                        ((i-1) * w, int(y1 * h)),
                        ((i) * w, int(y2 * h)),
                        (255, 255, 255), 2)

            canvas = np.pad(canvas, ((5, 5), (5, 5), (0, 0)))

            cv2.imshow('%.3f %.3f %.3f' % (self._K_P, self._K_I, self._K_D), canvas)
            cv2.waitKey(1)

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
