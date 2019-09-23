import  numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr


def generate_test_data(num_sample_images = 10, show_plot = 0):

    image_part_1 = np.array([[0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])

    image_part_2 = np.array([[0,0,0,1,1,1],[0,0,0,1,0,1],[0,0,0,1,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])

    image_part_3 = np.array(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0],
         ])

    image_part_4 = np.array(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0],
         ])

    if show_plot:
        plt.subplot(2, 2, 1)
        plt.imshow(image_part_1, cmap="gray")

        plt.subplot(2, 2, 2)
        plt.imshow(image_part_2, cmap="gray")

        plt.subplot(2, 2, 3)
        plt.imshow(image_part_3, cmap="gray")

        plt.subplot(2, 2, 4)
        plt.imshow(image_part_4, cmap="gray")
        plt.show()

    A = np.zeros((4,36))
    A[0,:] =np.ravel(image_part_1)
    A[1, :] = np.ravel(image_part_2)
    A[2, :] = np.ravel(image_part_3)
    A[3, :] = np.ravel(image_part_4)

    num_latent_image_features = A.shape[0]
    vector_image_size = A.shape[1]
    sigma_X = .5


    Z = np.round(npr.rand(num_sample_images, num_latent_image_features))

    X = np.matmul(Z,  A) + npr.randn(num_sample_images, vector_image_size) * sigma_X

    return X, Z.astype(int), A, sigma_X


    
if __name__ == '__main__':
    
    X, Z, A, sigma_X = generate_test_data(10)

    '''
    X = np.array([[1,1,1],[1,1,1]])
    Z = np.array([[1,0],[0,1]])
    A = np.array([[1, 0], [0, 1]])
    lp = logPX(X, Z, sigma_X, 1)
    print(lp)
    lpz = logPXZ(X, Z, sigma_X, 1, 4)
    print(lpz)
    '''
    
