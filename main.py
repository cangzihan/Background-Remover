import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def background_remove(image):
    scaling_factor = 0.05
    # copy one
    img = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    img_mini = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    pixel_list = img_mini.tolist()
    pixel_list = np.array([i for item in pixel_list for i in item])
    clustering = DBSCAN(eps=20, min_samples=10).fit(pixel_list)
    areas = []
    for i in set(clustering.labels_):
        areas.append(np.sum(clustering.labels_ == i))

    bg_index = areas.index(np.max(areas))

    labels = np.array(clustering.labels_).reshape(len(img_mini), len(img_mini[0]))
    mean_color = np.array([np.mean(img_mini[labels == bg_index][..., 0]),
                           np.mean(img_mini[labels == bg_index][..., 1]),
                           np.mean(img_mini[labels == bg_index][..., 2])])

    max_color = np.array([np.max(img_mini[labels == bg_index][..., 0]),
                           np.max(img_mini[labels == bg_index][..., 1]),
                           np.max(img_mini[labels == bg_index][..., 2])])

    min_color = np.array([np.min(img_mini[labels == bg_index][..., 0]),
                           np.min(img_mini[labels == bg_index][..., 1]),
                           np.min(img_mini[labels == bg_index][..., 2])])

    color_range = np.sqrt(np.sum(np.square(max_color - min_color))) / 2
    print("Background color:", mean_color)

    chazhi = np.sqrt(np.sum(np.square(img - mean_color), axis=2))

    img[chazhi <= color_range] = np.array([255, 255, 255])

    return img


if __name__ == '__main__':
    pic_path = r"test.jpg"
    img = cv2.imread(pic_path)
    import time
    t0 = time.time()
    img_new = background_remove(img)
    t = time.time() - t0
    print(t)

    cv2.imshow('Origin', img)
    cv2.imshow('Remove', img_new)
    cv2.waitKey()

