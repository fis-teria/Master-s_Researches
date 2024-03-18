import cv2
import numpy as np
import sys

left = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
right = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

d_left = cv2.cuda_GpuMat()
d_right = cv2.cuda_GpuMat()
d_left.upload(left);
d_right.upload(right);

sgm = cv2.cuda.createStereoSGM(0, 256)
d_disp = sgm.compute(d_left, d_right)
disp = d_disp.download()

disp = disp.astype(np.double) * 256 / (sgm.getNumDisparities() * cv2.StereoMatcher_DISP_SCALE)
disp_8u = disp.astype(np.uint8)
colored = cv2.applyColorMap(disp_8u, cv2.COLORMAP_JET)
colored[disp < 0] = np.array([0, 0, 0], dtype = np.uint8)
disp_8u[disp < 0] = 0

cv2.imshow('left', left)
cv2.imshow('disp', disp_8u)
cv2.imshow('colored', colored)
cv2.waitKey(0)