import matplotlib.pyplot as plt
from numpy import prod
import pandas as pd
import numpy as np
import cv2

def vec2img(vec, img_size=(40, 40)):
    return vec.reshape(img_size)

def img2vec(img):
    return img.reshape((prod(img.shape)))

def horizontally_flipped(img):
    img_flip = np.flip(img, axis=1)
    return img_flip

def split_img(img, img_size=(40, 40), flip=True):
    imgs = []
    n_rows, n_cols = img.shape[0], img.shape[1]
    img_height = img_size[0]
    img_width = img_size[1]
    ys = []
    for row in range(img_height, n_rows+1, img_height):
        user_id = 0
        for col in range(img_width, n_cols+1, img_width):
            row_last = row - img_height
            col_last = col - img_width
            img_crop = img[row_last:row, col_last:col]
            imgs.append(img_crop)
            ys.append(user_id)
            if flip:
                imgs.append(horizontally_flipped(img_crop))
                ys.append(user_id)
            user_id += 1
    return np.array(imgs), np.array(ys)

def weight_distance(w1, w2, eigenValues):
    eigenValues = eigenValues[:len(w1)]
    w1 = np.array(w1)
    w2 = np.array(w2)
    diff = (w1 - w2) / eigenValues**(1/2)
    return np.dot(diff, diff)


class EigenFace:
    def __init__(self):
        pass
    
    def fit(self, X, y, k=25):
        self.X = X # (num_data, vec_len)
        self.y = y # (num_data, )
        N, D = X.shape[0], X.shape[1]
        imgs_vec = X.T
        # Find mean vector
        mean_vector = X.mean(0).T
        self.mean_vector = mean_vector
        # plt.imshow(vec2img(mean_vector), cmap='gray'), plt.show() # Mean Face
        
        diff_imgs = imgs_vec - np.tile(np.array([mean_vector]).T, (1, N))
        T_trans_T = np.cov(diff_imgs.T)
        
        eigenValues, eigenVectors = np.linalg.eig(T_trans_T)
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        # First K eigenvector
        eigenValues = eigenValues[:k]
        eigenVectors = eigenVectors[:, 0:k]
        
        # Get EigenFace
        A = diff_imgs
        eigenVectors = np.matmul(A, eigenVectors)
        eigen_faces_vec = np.copy(eigenVectors)
        
        for i in range(k):
            eigen_faces_vec[:, i] =  eigenVectors[:,i] / np.linalg.norm(eigenVectors[:,i])
            
        eigen_faces_vec = eigen_faces_vec.T  # (K, vec_length)
        self.eigen_faces_vec = eigen_faces_vec
        self.eigen_values = eigenValues
    
    def show_eigen_face(self, k=25):
        k = min(k, len(self.eigen_faces_vec))
        plt.figure(figsize=(6, 7.5))
        n_row = int(k**(1/2))
        n_col = int(k**(1/2))
        for i in range(k):
            face = vec2img(self.eigen_faces_vec[i])
            plt.subplot(n_row ,n_col, i+1)
            plt.imshow(face, cmap = 'gray'), plt.xticks([]) , plt.yticks([])
            
    def reconstruction(self, img):
        k = self.eigen_faces_vec.shape[0]
        img_vec = img2vec(img)
        diff_vec = img_vec - self.mean_vector
        diff_face = vec2img(diff_vec)
        weights = []
        for i in range(k):
            weights.append(np.dot(diff_vec, self.eigen_faces_vec[i]))
        reconstruct_vec = np.zeros(self.mean_vector.shape)
        reconstruct_faces = []
        for i in range(k):
            reconstruct_vec = reconstruct_vec + weights[i] * self.eigen_faces_vec[i]
            reconstruct_face = vec2img(reconstruct_vec+mean_vector)
            reconstruct_faces.append(reconstruct_face)
        return reconstruct_faces
    
    def predict(self, img, n_eigenvec=10):
        img = img.copy()
        k = self.eigen_faces_vec.shape[0]
        
        # Find input weight
        img_vec = img2vec(img)
        diff_vec = img_vec - self.mean_vector
        diff_face = vec2img(diff_vec)
        input_weights = []
        for i in range(n_eigenvec):
            input_weights.append(np.dot(diff_vec, self.eigen_faces_vec[i]))
        
        best_match_img = None
        best_match_error = 1e+10
        best_y = 0
        # Compare with training data
        for x, y  in zip(self.X, self.y): # (num_data, vec_len)
            img = vec2img(x)
            img_vec = img2vec(img)
            diff_vec = img_vec - self.mean_vector
            diff_face = vec2img(diff_vec)
            target_weights = []
            for i in range(n_eigenvec):
                target_weights.append(np.dot(diff_vec, self.eigen_faces_vec[i]))
            error = weight_distance(input_weights, target_weights, self.eigen_values)
            if error < best_match_error:
                best_match_img = img.copy()
                best_match_error = error
                best_y = y
        return best_match_img, best_y