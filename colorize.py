"""
Credits: 
	1. https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
	2. http://richzhang.github.io/colorization/
	3. https://github.com/richzhang/colorization/
"""

# Import statements
import numpy as np
import cv2
import os

"""
Download the model files: 
	1. colorization_deploy_v2.prototxt:    https://github.com/richzhang/colorization/tree/caffe/colorization/models
	2. pts_in_hull.npy:					   https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
	3. colorization_release_v2.caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

"""

# Paths to load the model
DIR = r"E:\kì 4\CPV301\go"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Thay vì sử dụng argparse, bạn sẽ chỉ định đường dẫn đến ảnh trực tiếp tại đây:
image_path = r"E:\kì 4\CPV301\go\images\4.jpg"  # Đặt đường dẫn ảnh cố định

# Load the Model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv2.imread(image_path)

# Kiểm tra nếu không tìm thấy ảnh
if image is None:
    print(f"Error: Cannot load image at {image_path}")
    exit(1)

# Tiền xử lý hình ảnh
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Resize ảnh để khớp với kích thước đầu vào của mô hình
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Tô màu hình ảnh
print("Colorizing the image...")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# Kết hợp kênh L và ab lại
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Chuyển đổi từ không gian màu LAB sang BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

# Chuyển đổi về uint8
colorized = (255 * colorized).astype("uint8")

# Hiển thị ảnh gốc và ảnh đã tô màu
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
