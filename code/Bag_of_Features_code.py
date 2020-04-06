import cv2
import numpy as np
import pickle
from PA4_utils import load_image, load_image_gray
#import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def build_vocabulary(image_paths, vocab_size, which_descriptor='sift'):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
  -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary
  -   which_descirptor: The feature descriptor to use. i.e. 'sift' or 'surf'

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """
  # Load images from the training set. To save computation time, you don't
  # necessarily need to sample from all images, although it would be better
  # to do so. You can randomly sample the descriptors from each image to save
  # memory and speed up the clustering. Or you can simply call vl_dsift with
  # a large step size here, but a smaller step size in get_bags_of_sifts.
  #
  # For each loaded image, get some SIFT features. You don't have to get as
  # many SIFT features as you will in get_bags_of_sift, because you're only
  # trying to get a representative sample here.
  #
  # Once you have tens of thousands of SIFT features from many training
  # images, cluster them with kmeans. The resulting centroids are now your
  # visual word vocabulary.

  dim = 128      # length of the SIFT descriptors that you are going to compute.
  #vocab = np.zeros((vocab_size,dim))


  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  #raise NotImplementedError('`build_vocabulary` function in ' +
  #      '`Bag_of_Features_code.py` needs to be implemented')

  descriptors = []
  for path in image_paths:
      img = cv2.imread(path)
      if which_descriptor == 'sift':
          sift_or_surf = cv2.xfeatures2d.SIFT_create(400)
          _, des = sift_or_surf.detectAndCompute(img,None)
          descriptors.append(des)
      elif which_descriptor == 'surf':
          sift_or_surf = cv2.xfeatures2d.SURF_create(10,extended=True)
          _, des = sift_or_surf.detectAndCompute(img,None)
          descriptors.append(des[:400,:])
  descriptors = np.concatenate(descriptors, axis=0).astype('float32')
  vocab_kmeans = KMeans(n_clusters=vocab_size).fit(descriptors)
  vocab = vocab_kmeans.cluster_centers_

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return vocab

def get_bags_of_sifts(image_paths, vocab_filename, which_descriptor='sift'):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.
  -   which_descirptor: The feature descriptor to use. i.e. 'sift' or 'surf'

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)

  # dummy features variable
  feats = []

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  #raise NotImplementedError('`get_bags_of_sifts` function in ' +
  #      '`Bag_of_Features_code.py` needs to be implemented')
  for path in image_paths:
      img = cv2.imread(path)
      if which_descriptor == 'sift':
          sift_or_surf = cv2.xfeatures2d.SIFT_create(400) # Gives only 400 keypoints
          _, des = sift_or_surf.detectAndCompute(img,None)
          dist = cdist(vocab, des[:400,:], metric='euclidean')
      elif which_descriptor == 'surf':
          sift_or_surf = cv2.xfeatures2d.SURF_create(10,extended=True) # extended=True gives 128 dim'l vector. 10 is value of hessianthreshold.
          _, des = sift_or_surf.detectAndCompute(img,None)
          dist = cdist(vocab, des[:400,:], metric='euclidean') # Takes top 400 keypoints
      indexes = np.argmin(dist, axis=0)
      hist, _ = np.histogram(indexes, bins=len(vocab))
      normal_hist = hist / np.sum(hist)   # L1 normalize
      feats.append(normal_hist)
  feats = np.asarray(feats)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean', n_neighbors=1):
  """
  This function will predict the category for every test image by finding
  the training image with most similar features. Instead of 1 nearest
  neighbor, you can vote based on k nearest neighbors which will increase
  performance (although you need to pick a reasonable value for k).

  Useful functions:
  -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
          -  X is a N x d numpy array of d-dimensional features arranged along
          N rows
          -  Y is a M x d numpy array of d-dimensional features arranged along
          N rows
          -  D is a N x M numpy array where d(i, j) is the distance between row
          i of X and row j of Y

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
          well for histograms
  -   n_neighbors: No. of nearest neighbors to be considered for classification

  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  #test_labels = []

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  #raise NotImplementedError('`nearest_neighbor_classify` function in ' +
  #      '`Bag_of_Features_code.py` needs to be implemented')

  def chisq_dist(a,b):
      return np.sum(pow((a-b),2) / (a+b+1)) # +1 so that it doesn't divide by 0 values.

  le = LabelEncoder()
  y_train = le.fit_transform(train_labels)

  if metric == 'euclidean':
      knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric = 'euclidean')
  elif metric == 'chi2':
      knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric = chisq_dist)
  knn.fit(train_image_feats,y_train)
  y_pred = knn.predict(test_image_feats)

  test_labels = list(le.inverse_transform(y_pred))
  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return  test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats, C=1.0):
  """
  This function will train a linear SVM for every category (i.e. one vs all)
  and then use the learned linear classifiers to predict the category of
  every test image. Every test feature will be evaluated with all 15 SVMs
  and the most confident SVM will "win". Confidence, or distance from the
  margin, is W*X + B where '*' is the inner product or dot product and W and
  B are the learned hyperplane parameters.

  Useful functions:
  -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  -   svm.fit(X, y)
  -   set(l)

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   C: Parameter that controls extent of regularization.
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  # categories
  categories = list(set(train_labels))

  # construct 1 vs all SVMs for each category
  #svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5) for cat in categories}

  #test_labels = []

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  #raise NotImplementedError('`svm_classify` function in ' +
  #      '`Bag_of_Features_code.py` needs to be implemented')

  le = LabelEncoder()
  y_train = le.fit_transform(train_labels)
  svc = LinearSVC(C=C, max_iter=2000)
  svc.fit(train_image_feats,y_train)
  y_pred = svc.predict(test_image_feats)
  test_labels = list(le.inverse_transform(y_pred))



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return test_labels



def show_interest_points(img, X, Y):
    """
    Visualized interest points on an image with random colors

    Args:
    - img: A numpy array of shape (M,N,C)
    - X: A numpy array of shape (k,) containing x-locations of interest points
    - Y: A numpy array of shape (k,) containing y-locations of interest points

    Returns:
    - newImg: A numpy array of shape (M,N,C) showing the original image with
            colored circles at keypoints plotted on top of it
    """
    newImg = img.copy()
    for x, y in zip(X.astype(int), Y.astype(int)):
        cur_color = np.random.rand(3)
        newImg = cv2.circle(newImg, (x, y), 10, cur_color, -1, cv2.LINE_AA)

    return newImg


def tfidf_get_bags_of_sifts(image_paths, vocab_filename, which_descriptor='sift'):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.
  -   which_descirptor: The feature descriptor to use. i.e. 'sift' or 'surf'

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)

  # dummy features variable
  feats = []
  idf_matrix = []

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  #raise NotImplementedError('`get_bags_of_sifts` function in ' +
  #      '`Bag_of_Features_code.py` needs to be implemented')
  for path in image_paths:
      img = cv2.imread(path)
      if which_descriptor == 'sift':
          sift_or_surf = cv2.xfeatures2d.SIFT_create(400) # Gives only 400 keypoints
          _, des = sift_or_surf.detectAndCompute(img,None)
          dist = cdist(vocab, des[:400,:], metric='euclidean')
      elif which_descriptor == 'surf':
          sift_or_surf = cv2.xfeatures2d.SURF_create(10,extended=True) # extended=True gives 128 dim'l vector. 10 is value of hessianthreshold.
          _, des = sift_or_surf.detectAndCompute(img,None)
          dist = cdist(vocab, des[:400,:], metric='euclidean') # Takes top 400 keypoints
      indexes = np.argmin(dist, axis=0)
      hist, _ = np.histogram(indexes, bins=len(vocab))
      idf_idxs = np.argwhere(hist)
      temp = np.zeros((hist.shape[0],),dtype='int')
      temp[idxs_idf] = 1                # temp array contains vector of 1's and 0's for each visual word corresponding to the image. 1/0 -> Visual word present/absent.
      idf_matrix.append(temp)
      feats.append(hist)
  feats = np.asarray(feats)
  idf_matrix = np.asarray(idf_matrix)
  idf_words = np.log(len(image_paths) / (np.sum(idf_matrix,axis=0)+1)) # idf_words is an array of length of visual words, with IDF value of each visual word.
  feats = feats * idf_words # TF-IDF
  feats = feats / np.sum(train_image_feats,axis=1).reshape(-1,1) # L1 normalize

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return feats

