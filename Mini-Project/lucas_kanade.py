import numpy as np
import cv2

def optical_flow(im1, im2, pts, win):
    # Initialize kernels for convolution
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    # Convolve to get image derivates
    fx = cv2.filter2D(im1, -1, kernel_x)
    fy = cv2.filter2D(im1, -1, kernel_y)
    ft = cv2.filter2D(im2, -1, kernel_t) - cv2.filter2D(im1, -1, kernel_t)

    op_flow = np.zeros(pts.shape)
    count = 0

    # For each feature point
    for p in pts:
        # Get coordinates
        j, i = p.ravel()
        j, i = int(j), int(i)
        
        # Get derivatives for window
        I_x = fx[i-win:i+win+1, j-win:j+win+1].flatten()
        I_y = fy[i-win:i+win+1, j-win:j+win+1].flatten()
        I_t = ft[i-win:i+win+1, j-win:j+win+1].flatten()

        # Make A and b (S and T)
        b = np.reshape(I_t, (I_t.shape[0],1))
        A = np.vstack((I_x, I_y)).T

        # Calculate u and v
        U = np.matmul(np.linalg.pinv(A), b)

        op_flow[count, 0, 0] = U[0, 0]
        op_flow[count, 0, 1] = U[1, 0]
        count += 1
 
    return op_flow