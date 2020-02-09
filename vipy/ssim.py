from scipy.signal import gaussian, convolve2d
import numpy as np


class SSIM(object):
    """Structural similarity (SSIM) index """
    """Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, "Image quality assessment: from error visibility to structural similarity". IEEE Transactions on Image Processing. 13 (4): 600–612"""

    def __init__(self, do_alignment=True, min_matches_for_alignment=10, num_matches_for_alignment=500, K1=0.01, K2=0.03):
        try_import('cv2', 'opencv-python'); import cv2  # optional

        self.do_alignment = do_alignment
        self.min_matches_for_alignment = min_matches_for_alignment
        self.num_matches_for_alignment = num_matches_for_alignment
        self.K1 = K1
        self.K2 = K2

    def __repr__(self):
        return str('<SSIM: do_alignment=%s, min_matches_for_alignment=%d, num_matches_for_alignment=%d, K1=%f, K2=%f>' % (str(self.do_alignment), self.min_matches_for_alignment, self.num_matches_for_alignment, self.K1, self.K2))

    def match(self, img1, img2):
        """Return a set of matching points in img1 and img2 in the form suitable for homography estimation"""

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        # Match descriptors.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x:x.distance)[:self.num_matches_for_alignment]

        img1_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        img2_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        return (img1_pts, img2_pts)

    def warp(self, src_pts, dst_pts, im_src):
        """Warp an image im_src with points src_pts to align with dst_pts"""
        if src_pts.shape[0] < self.min_matches_for_alignment:
            raise ValueError('Invalid number of inliers')
        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

    def align(self, img1, img2):
        """Return an image which is the warped version of img1 that aligns with img2"""
        (p1, p2) = self.match(img1, img2)
        return self.warp(p1, p2, img1)

    def rgb2gray(self, I):
        """Convert RGB image to grayscale; accesory function"""
        R = I[:,:,0]
        G = I[:,:,1]
        B = I[:,:,2]
        return 0.299 * R + 0.587 * G + 0.114 * B

    def similarity(self, I1, I2, returnMap=True):
        """Compute the Structural Similarity Index (SSIM) score of two images
        Inputs:
        1) I1, image array
        2) I2, image array
        3) K1, float (optional, default=0.01)
        - constant
        4) K2, float (optional, default=0.03)
        - constant
        Outputs:
        1) out; float
        - SSIM score
        2) ssim_map; 2-D image array
        - SSIM map"""

        I1 = self.rgb2gray(I1) if I1.ndim == 3 else I1
        I2 = self.rgb2gray(I2) if I2.ndim == 3 else I2

        C1 = np.power(self.K1 * 255,2)
        C2 = np.power(self.K2 * 255,2)

        w = gaussian(11,1.5)
        f = np.zeros((11,11))
        for k in range(len(w)):
            for k2 in range(len(w)):
                f[k,k2] = np.multiply(w[k],w[k2])
        f = np.true_divide(f,np.sum(f))

        ux = convolve2d(I1,f,mode='same')
        uy = convolve2d(I2,f,mode='same')

        # Compute SSIM constants
        ux_sq = np.power(ux,2)
        uy_sq = np.power(uy,2)
        ux_uy = np.multiply(ux,uy)

        sig_x = convolve2d(np.power(I1,2),f,mode='same') - ux_sq
        sig_y = convolve2d(np.power(I2,2),f,mode='same') - uy_sq
        sig_xy = convolve2d(np.multiply(I1,I2),f,mode='same') - ux_uy

        # Core SSIM Equation
        ssim_map = np.divide(np.multiply(2 * ux_uy + C1, 2 * sig_xy + C2),
                             np.multiply(ux_sq + uy_sq + C1, sig_x + sig_y + C2))

        out = np.mean(ssim_map)

        return (out, ssim_map) if returnMap else out

    def ssim(self, im_reference, im_degraded):
        """Return structural similarity score when aligning im_degraded to im_reference"""
        im_degraded_aligned = self.align(im_degraded, im_reference) if self.do_alignment else im_degraded
        return self.similarity(im_degraded_aligned, im_reference, returnMap=False)


def demo(imfile):
    """Synthetically rotate an image by 10 degrees, and compute structural similarity with and without alignment, return images"""
    img1 = cv2.imread(imfile, 0)

    (num_rows, num_cols) = img1.shape[:2]
    R = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 10, 1)
    img1_rotation = cv2.warpAffine(img1, R, (num_cols, num_rows))
    img2 = img1_rotation

    print('Structural similarity score (aligned): %f' % SSIM(do_alignment=True).ssim(img1, img2))
    print('Structural similarity score (unaligned): %f' % SSIM(do_alignment=False).ssim(img1, img2))

    return (img1, img2)
