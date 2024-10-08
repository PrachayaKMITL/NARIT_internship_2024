import numpy as np
import warnings

__all__ = ['compute_glcm','compute_properties','statistical']

def compute_glcm(image, distances, angles, symmetric=False, normalized=False):
    """
    Compute the GLCM matrix for a given image.

    Parameters:
        image (2D array): Grayscale image (pixel values must be between 0 and 255).
        distances (list of int): List of pixel pair distances.
        angles (list of float): List of angles in radians (e.g., 0, np.pi/4, np.pi/2, 3*np.pi/4).
        symmetric (bool): If True, the GLCM will be symmetric.
        normalized (bool): If True, the GLCM will be normalized.

    Returns:
        glcm (ndarray): The GLCM matrix of shape (256, 256, len(distances), len(angles)).
    """
    
    # Initialize the GLCM matrix with zeros
    glcm = np.zeros((256, 256, len(distances), len(angles)), dtype=np.float64)
    
    # Get image dimensions and ensure it's grayscale
    if image.ndim == 3:  # Color image, convert to grayscale
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    # Ensure the pixel values are integers and in the valid range
    image = np.clip(image, 0, 255).astype(int)  # Clip values to the valid range and convert to int
    rows, cols = image.shape
    
    for d_idx, distance in enumerate(distances):
        for a_idx, angle in enumerate(angles):
            # Calculate the row and column offsets based on the angle and distance
            row_offset = int(round(np.sin(angle) * distance))
            col_offset = int(round(np.cos(angle) * distance))
            
            for i in range(rows):
                for j in range(cols):
                    current_pixel = image[i, j]
                    
                    # Calculate the neighboring pixel location
                    row_neighbor = i + row_offset
                    col_neighbor = j + col_offset
                    
                    # Ensure the neighbor is within bounds
                    if 0 <= row_neighbor < rows and 0 <= col_neighbor < cols:
                        neighbor_pixel = image[row_neighbor, col_neighbor]
                        
                        # Increment the corresponding GLCM value
                        glcm[current_pixel, neighbor_pixel, d_idx, a_idx] += 1

                        # For symmetry, also add the reverse pair
                        if symmetric:
                            glcm[neighbor_pixel, current_pixel, d_idx, a_idx] += 1

    # Normalize GLCM to get probabilities
    if normalized:
        glcm_sum = glcm.sum(axis=(0, 1), keepdims=True) + 1e-6  # Prevent division by zero
        glcm /= glcm_sum
    
    return glcm

def compute_properties(P, prop='contrast'):
    """Calculate texture properties of a GLCM.

    Compute a feature of a gray level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:

    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]
    - 'mean': :math:`\\sum_{i=0}^{levels-1} i*P_{i}`
    - 'variance': :math:`\\sum_{i=0}^{levels-1} P_{i}*(i-mean)^2`
    - 'std': :math:`\\sqrt{variance}`
    - 'entropy': :math:`\\sum_{i,j=0}^{levels-1} -P_{i,j}*log(P_{i,j})`

    Each GLCM is normalized to have a sum of 1 before the computation of
    texture properties.

    .. versionchanged:: 0.19
           `greycoprops` was renamed to `graycoprops` in 0.19.

    Parameters
    ----------
    P : ndarray
        Input array. `P` is the gray-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that gray-level j
        occurs at a distance d and at an angle theta from
        gray-level i.
    prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM', 'mean', 'variance', 'std', 'entropy'}, optional
        The property of the GLCM to compute. The default is 'contrast'.

    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial v. 1.0 through 3.0.
           The GLCM Tutorial Home Page,
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`

    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = graycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = graycoprops(g, 'contrast')
    >>> contrast
    array([[0.58333333, 1.        ],
           [1.25      , 2.75      ]])

    """

    def glcm_mean():
        I = np.arange(num_level).reshape((num_level, 1, 1, 1))
        mean = np.sum(I * P, axis=(0, 1))
        return I, mean

    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1.0 / (1.0 + (I - J) ** 2)
    elif prop in ['ASM', 'energy', 'correlation', 'entropy', 'variance', 'mean', 'std']:
        pass
    else:
        raise ValueError(f'{prop} is an invalid property')

    # compute property for each GLCM
    if prop == 'energy':
        asm = np.sum(P**2, axis=(0, 1))
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.sum(P**2, axis=(0, 1))
    elif prop == 'mean':
        _, results = glcm_mean()
    elif prop == 'variance':
        I, mean = glcm_mean()
        results = np.sum(P * ((I - mean) ** 2), axis=(0, 1))
    elif prop == 'std':
        I, mean = glcm_mean()
        var = np.sum(P * ((I - mean) ** 2), axis=(0, 1))
        results = np.sqrt(var)
    elif prop == 'entropy':
        ln = -np.log(P, where=(P != 0), out=np.zeros_like(P))
        results = np.sum(P * ln, axis=(0, 1))

    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.sum(I * P, axis=(0, 1))
        diff_j = J - np.sum(J * P, axis=(0, 1))

        std_i = np.sqrt(np.sum(P * (diff_i) ** 2, axis=(0, 1)))
        std_j = np.sqrt(np.sum(P * (diff_j) ** 2, axis=(0, 1)))
        cov = np.sum(P * (diff_i * diff_j), axis=(0, 1))

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = ~mask_0
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
    elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum(P * weights, axis=(0, 1))

    return results

def statistical(img):
    '''Statsistical value 
    derived from img and use statistical method to 
    extract information
    these value comprised of 
        - Mean_R : Mean of red channel value in an image
        - Mean_B : Mean of Blue channel value in an image
        - Different R-B : Mean of the different of an image in Blue and Red channel
        - std : Standard deviation of the Blue channel in an image
    return: 
        statistic : DICT
    '''
    # Check for valid image dimensions
    if img.ndim != 3:
        raise ValueError("Invalid dimension (check image color channel)")

    # Extract RGB channels
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Calculate statistics
    Mean_R = np.mean(R)
    Mean_B = np.mean(B)

    #Calculate standard deviation
    Std_B = np.std(B)
    
    # Calculate differences
    diff_RB = np.mean(R - B)    
    # Concatenate all statistics into a single array
    stats = {
        "Mean R channel" : Mean_R,
        "Mean_B channel" : Mean_B,
        "Std B  channel" : Std_B,
        "Different R-B"  : diff_RB
    }

    return stats

