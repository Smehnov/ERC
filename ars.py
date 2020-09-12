import cv2.aruco as aruco
import numpy as np
import cv2

aruco_dict = aruco.custom_dictionary(0, 7)
aruco_dict.bytesList = np.zeros(shape=(16, 7, 4), dtype=np.uint8)

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 1, 0],
               [1, 1, 1, 0, 1]], dtype=np.uint8)
aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 1, 0],
               [1, 0, 1, 1, 0]], dtype=np.uint8)
aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 1, 1, 1, 1],
               [1, 0, 1, 0, 0]], dtype=np.uint8)
aruco_dict.bytesList[3] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0]], dtype=np.uint8)
aruco_dict.bytesList[4] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 1, 1],
               [0, 1, 1, 0, 0]], dtype=np.uint8)
aruco_dict.bytesList[5] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[6] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 1, 1, 1, 0],
               [0, 0, 1, 0, 1]], dtype=np.uint8)
aruco_dict.bytesList[7] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 0, 1],
               [1, 1, 1, 1, 0]], dtype=np.uint8)
aruco_dict.bytesList[8] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 1, 1, 0, 0],
               [1, 1, 1, 0, 0]], dtype=np.uint8)
aruco_dict.bytesList[9] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 1, 1, 0, 0],
               [1, 0, 1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[10] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 0, 1]], dtype=np.uint8)
aruco_dict.bytesList[11] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 0, 0],
               [0, 1, 1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[12] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 1, 1, 0, 1],
               [0, 1, 1, 0, 1]], dtype=np.uint8)
aruco_dict.bytesList[13] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 1, 1, 0, 1],
               [0, 0, 1, 1, 0]], dtype=np.uint8)
aruco_dict.bytesList[14] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]], dtype=np.uint8)
aruco_dict.bytesList[15] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))
