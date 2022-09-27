import vipy
from scipy.signal import convolve2d
from vipy.math import gaussian
import numpy as np
from vipy.globals import print
from vipy.util import try_import
try_import('cv2', 'opencv-python'); import cv2 


class SSIM(object):
    """Structural similarity (SSIM) index """
    """Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, "Image quality assessment: from error visibility to structural similarity". IEEE Transactions on Image Processing. 13 (4): 600–612"""

    def __init__(self, do_alignment=True, min_matches_for_alignment=10, num_matches_for_alignment=500, K1=0.01, K2=0.03):
        self.do_alignment = do_alignment
        self.min_matches_for_alignment = min_matches_for_alignment
        self.num_matches_for_alignment = num_matches_for_alignment
        self.K1 = K1
        self.K2 = K2

    def __repr__(self):
        return str('<vipy.ssim: do_alignment=%s, min_matches_for_alignment=%d, num_matches_for_alignment=%d, K1=%f, K2=%f>' % (str(self.do_alignment), self.min_matches_for_alignment, self.num_matches_for_alignment, self.K1, self.K2))

    def match(self, img1, img2):
        """Return a set of matching points in img1 (MxN uint8 numpy) and img2 (MxN uint8 numpy) in the form suitable for homography estimation"""

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
        """Return an image which is the warped version of img1 (MxN uint8 numpy) that aligns with img2 (MxN uint8 numpy)"""
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

    
    def ssim(self, im_reference, im_degraded, returnAligned=False):
        """Return structural similarity score when aligning im_degraded to im_reference

        >>> (ssim, im_aligned) = vipy.ssim.SSIM(do_alignment=True).ssim(vipy.image.squareowl(), vipy.image.squareowl().rotate(0.01), returnAligned=True)
        >>> print(ssim)
        >>> im_aligned.show(figure=1)
        >>> vipy.image.squareowl().rotate(0.01).show(figure=2)
        
        """
        assert isinstance(im_reference, np.ndarray) or isinstance(im_reference, vipy.image.Image)
        assert isinstance(im_degraded, np.ndarray) or isinstance(im_degraded, vipy.image.Image)
        
        img_degraded = im_degraded.lum().numpy() if isinstance(im_degraded, vipy.image.Image) else im_degraded
        img_reference = im_reference.lum().numpy() if isinstance(im_reference, vipy.image.Image) else im_reference
        
        img_degraded_aligned = self.align(img_degraded, img_reference) if self.do_alignment else im_degraded
        ssim = self.similarity(img_degraded_aligned, img_reference, returnMap=False)
        return (ssim, vipy.image.Image(array=img_degraded_aligned, colorspace='lum')) if returnAligned else ssim


def demo(im=None):
    """Synthetically rotate an image by 4 degrees, and compute structural similarity with and without alignment, return images
    
    >>> (image, degraded_image, aligned_image) = vipy.ssim.demo(vipy.image.Image(filename='/path/to/image.jpg')))
    
    """
    assert im is None or isinstance(im, vipy.image.Image)
    im = vipy.image.squareowl() if im is None else im
    
    # Synthetic degradation: 1-channel uint8
    (im, im_degraded) = (im.lum(), im.clone().rotate(4*(np.pi/180.0)).lum())
    
    # SSIM
    (ssim_aligned, im_aligned) = SSIM(do_alignment=True).ssim(im.numpy(), im_degraded.numpy(), returnAligned=True)
    (ssim_unaligned) = SSIM(do_alignment=False).ssim(im.numpy(), im_degraded.numpy())    
    print('Structural similarity score (aligned): %f' % ssim_aligned)
    print('Structural similarity score (unaligned): %f' % ssim_unaligned)
    return (im.show(figure=1),
            im_degraded.show(figure=2),
            im_aligned.show(figure=3))
