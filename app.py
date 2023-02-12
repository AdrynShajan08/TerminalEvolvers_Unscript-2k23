from flask import Flask,render_template,redirect,request
from werkzeug.utils import secure_filename
import os
UPLOAD_FOLDER = '/uploads/'
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

import collections
from itertools import chain
# from urllib3 import request
import pickle
import cv2 as cv
import numpy as np
import pandas as pd

import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize

import matplotlib.pyplot as plt

import skimage.io
import skimage.transform

import cv2

from libsvm import svmutil

def normalize_kernel(kernel):
    return kernel / np.sum(kernel)

def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n/2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(gaussian_kernel)

def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')
def local_deviation(image, local_mean, kernel):
    "Vectorized approximation of local deviation"
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def calculate_mscn_coefficients(image, kernel_size=6, sigma=7 / 6):
    C = 1 / 255
    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(image, kernel, 'same')
    local_var = local_deviation(image, local_mean, kernel)

    return (image - local_mean) / (local_var + C)


def generalized_gaussian_dist(x, alpha, sigma):
    beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))

    coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
    return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)
def calculate_pair_product_coefficients(mscn_coefficients):
    return collections.OrderedDict({
        'mscn': mscn_coefficients,
        'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
        'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
        'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
        'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
    })


def asymmetric_generalized_gaussian(x, nu, sigma_l, sigma_r):
    def beta(sigma):
        return sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))

    coefficient = nu / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu))
    f = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu)

    return np.where(x < 0, f(-x, sigma_l), f(x, sigma_r))


def asymmetric_generalized_gaussian_fit(x):
    def estimate_phi(alpha):
        numerator = special.gamma(2 / alpha) ** 2
        denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
        return numerator / denominator

    def estimate_r_hat(x):
        size = np.prod(x.shape)
        return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

    def estimate_R_hat(r_hat, gamma):
        numerator = (gamma ** 3 + 1) * (gamma + 1)
        denominator = (gamma ** 2 + 1) ** 2
        return r_hat * numerator / denominator

    def mean_squares_sum(x, filter=lambda z: z == z):
        filtered_values = x[filter(x)]
        squares_sum = np.sum(filtered_values ** 2)
        return squares_sum / ((filtered_values.shape))

    def estimate_gamma(x):
        left_squares = mean_squares_sum(x, lambda z: z < 0)
        right_squares = mean_squares_sum(x, lambda z: z >= 0)

        return np.sqrt(left_squares) / np.sqrt(right_squares)

    def estimate_alpha(x):
        r_hat = estimate_r_hat(x)
        gamma = estimate_gamma(x)
        R_hat = estimate_R_hat(r_hat, gamma)

        solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x

        return solution[0]

    def estimate_sigma(x, alpha, filter=lambda z: z < 0):
        return np.sqrt(mean_squares_sum(x, filter))

    def estimate_mean(alpha, sigma_l, sigma_r):
        return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))

    alpha = estimate_alpha(x)
    sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
    sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)

    constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    mean = estimate_mean(alpha, sigma_l, sigma_r)

    return alpha, mean, sigma_l, sigma_r


def calculate_brisque_features(image, kernel_size=7, sigma=7 / 6):
    def calculate_features(coefficients_name, coefficients, accum=np.array([])):
        alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)

        if coefficients_name == 'mscn':
            var = (sigma_l ** 2 + sigma_r ** 2) / 2
            return [alpha, var]

        return [alpha, mean, sigma_l ** 2, sigma_r ** 2]

    mscn_coefficients = calculate_mscn_coefficients(image, kernel_size, sigma)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)

    features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
    flatten_features = list(chain.from_iterable(features))
    return np.array(flatten_features)
def load_image(url):
    image_stream = request.urlopen(url)
    return skimage.io.imread(image_stream, plugin='pil')

def plot_histogram(x, label):
    n, bins = np.histogram(x.ravel(), bins=50)
    n = n / np.max(n)
    plt.plot(bins[:-1], n, label=label, marker='o')


def scale_features(features):
    with open('normalize.pickle', 'rb') as handle:
        scale_params = pickle.load(handle)

    min_ = np.array(scale_params['min_'])
    max_ = np.array(scale_params['max_'])

    return -1 + (2.0 / (max_ - min_) * (features - min_))


def calculate_image_quality_score(brisque_features):
    model = svmutil.svm_load_model('brisque_svm.txt')
    scaled_brisque_features = scale_features(brisque_features)

    x, idx = svmutil.gen_svm_nodearray(
        scaled_brisque_features,
        isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))

    nr_classifier = 1
    prob_estimates = (svmutil.c_double * nr_classifier)()

    return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)
from skimage import io
def brisque(image):
    k=image
    print(k)
#     image = io.imread(k)
    image=cv2.imread(k)
#     gray_image = skimage.color.rgb2gray(image)
    gray_image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     mscn_coefficients = calculate_mscn_coefficients(gray_image, 7, 7/6)
#     coefficients = calculate_pair_product_coefficients(mscn_coefficients)
    brisque_features = calculate_brisque_features(gray_image, kernel_size=7, sigma=7/6)
    downscaled_image = cv2.resize(gray_image, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
    downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)
    brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
    return calculate_image_quality_score(brisque_features)


from sklearn.preprocessing import MinMaxScaler


import cv2

images_path = "data/"


def preprocessing(imager):
    x = 350
    details = []
    threshold_min = 0.1
    threshold_max = 100.0
    #     def brightness(image):
    # path = images_path + imager
    #
    # print(path)
    image=cv2.imread(imager)

    image = cv2.resize(image, (x, x))
    image2 = image.copy()
    L, A, B = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)
    # Return True if mean is greater than thresh else False
    details.append(np.mean(L))
    #     def calculate_contrast(image):

    #     img.resize((350,350))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the standard deviation of the pixel intensities
    contrast = cv2.norm(gray, cv2.NORM_L2) / np.sqrt(gray.size)

    # Normalize the contrast value to a range of [0, 1]
    normalized_contrast = contrast / 255.0
    details.append(normalized_contrast)

    # def get_blurrness_score(image, ):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    fm = np.interp(fm, [threshold_min, threshold_max], [1, 100])
    details.append(fm)

    # def noisef(image):

    image = cv.cvtColor(image2, cv.COLOR_BGR2HSV)

    s = cv.calcHist([image], [1], None, [256], [0, 256])

    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    ##### Just for visualization and debug; remove in final

    # Percentage threshold; above: valid image, below: noise
    #     s_thr = 0.5
    details.append(s_perc)

    # def text_to_background(image):
    # Load an image
    #     image = cv2.imread("black.jpg")
    #     img.resize((350,350))
    # Convert the image to grayscale
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to separate the text and background
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Calculate the total number of pixels in the image
    total_pixels = binary.shape[0] * binary.shape[1]

    # Calculate the number of text pixels
    text_pixels = np.count_nonzero(binary)

    # Calculate the number of background pixels
    background_pixels = total_pixels - text_pixels
    if background_pixels == 0:
        return None
    # Calculate the text-to-background ratio
    text_to_background_ratio = text_pixels / background_pixels
    details.append(text_to_background_ratio)
    # def getDimensions(filename):
    details.append(image.shape[0])
    details.append(image.shape[1])
    #     return img_size
    # def find_artifacts(img):
    # Load the image in grayscale
    #     path=images_path+img
    img = cv2.imread(imager, 0)
    # img2 = cv2.imread('test.jpg', 0)
    img = cv2.resize(img, (x, x))
    threshold_value = 100
    max_value = 255
    ret, binary_image = cv2.threshold(img, threshold_value, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary_image = gsray <128
    # image = cv2.multiply(img, 1.5)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    number_of_white_pix = np.sum(img == 255)
    number_of_black_pix = np.sum(img == 0)

    number_of_white_pix1 = np.sum(binary_image == 255)
    number_of_black_pix1 = np.sum(binary_image == 0)
    height, width = binary_image.shape
    artificat_presence = number_of_black_pix1 / (height * width)
    details.append(artificat_presence)
    #     print(details)
    return details


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/',methods=['GET','POST'])
def home():
    if request.method=="POST":
            # check if the post request has the file part

        file = request.files['file']
        # if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save("uploads/"+filename)
        res=brisque(f'uploads/{filename}')
        ans=None
        if res > 105:
            ans = "Very Bad Quality"
        elif 95 < res < 105:
            ans = "Bad Quality"
        elif 85 < res < 95:
            ans = "Good Quality"
        elif 85 > res:
            ans = "Very Good Quality"
        print(ans)
        features=preprocessing(f'uploads/{filename}')

        return render_template('home.html',rest=True,ans=ans,features=features)
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/teachers')
def teachers():
    return render_template('teachers.html')

@app.route('/course')
def course():
    return render_template('course-1.html')

if __name__=="__main__":
    app.run(debug=True)