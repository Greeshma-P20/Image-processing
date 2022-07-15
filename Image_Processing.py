# ImageFilter for using filter() function
from PIL import Image, ImageFilter
import cv2 # pip install opencv-python
import numpy as np 
import matplotlib.pyplot as plt
import pywt
import pywt.data
from skimage.io import imread
from skimage.color import rgb2gray

def Gaussian_Filter(): # Apply a Gauss filter to an image
    
    # Opening the image 
    # (R prefixed to string in order to deal with '\' in paths)
    image = Image.open(r"E:\sem 3\spyder practise\Project\Gaussian_Filter_Image.png")
      
    # Blurring image by sending the ImageFilter.
    # GaussianBlur predefined kernel argument
    image = image.filter(ImageFilter.GaussianBlur)
      
    # Displaying the image
    image.save('Comlete_Blurred_Image.png')

    # Opening the image 
    # (R prefixed to string in order to deal with '\' in paths)
    image = Image.open(r"E:\sem 3\spyder practise\Project\Gaussian_Filter_Image.png")
        
    # Cropping the image 
    smol_image = image.crop((0, 0, 1600, 1600))
      
    # Blurring on the cropped image
    blurred_image = smol_image.filter(ImageFilter.GaussianBlur)
      
    # Pasting the blurred image on the original image
    image.paste(blurred_image, (0,0))
      
    # Displaying the image
    image.save('Blurring_Small_Region.png')


def Morphological_Operations():
    
    # return video from the first webcam on your computer. 
    screenRead = cv2.VideoCapture(0)
     
    # loop runs if capturing has been initialized.
    while(1):
        # reads frames from a camera
        _, image = screenRead.read()
         
        # Converts to HSV color space, OCV reads colors as BGR
        # frame is converted to hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
         
        # defining the range of masking
        blue1 = np.array([110, 50, 50])
        blue2 = np.array([130, 255, 255])
         
        # initializing the mask to be
        # convoluted over input image
        mask = cv2.inRange(hsv, blue1, blue2)
     
        # passing the bitwise_and over
        # each pixel convoluted
        cv2.bitwise_and(image, image, mask = mask)
         
        # defining the kernel i.e. Structuring element
        kernel = np.ones((5, 5), np.uint8)
         
        # defining the opening function
        # over the image and structuring element
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # The mask and opening operation
        # is shown in the window
        cv2.imshow('Mask', mask)
        cv2.imshow('Opening', opening)
         
        # Wait for 'a' key to stop the program
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
     
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()
     
    # Close the window / Release webcam
    screenRead.release()


def Fourier_Tranform():
    
    #load the image
    dark_image = imread('Fourier_Transform_Image.png')

    #image into greyscale
    dark_image_grey = rgb2gray(dark_image)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(dark_image_grey, cmap='gray')
    plt.title('Greyscale Image')

    #use the fft function found in Skimage
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray');
    plt.title('Fourier Transformation Image')

    #Fourier Transform Vertical Masked Image
    def fourier_masker_ver(image, i):
        f_size = 15
        dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(rgb2gray(image)))
        dark_image_grey_fourier[:225, 235:240] = i
        dark_image_grey_fourier[-225:,235:240] = i
        fig, ax = plt.subplots(1,3,figsize=(15,15))
        ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
        ax[0].set_title('Masked Fourier', fontsize = f_size)
        ax[1].imshow(rgb2gray(image), cmap = 'gray')
        ax[1].set_title('Greyscale Image', fontsize = f_size);
        ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), 
                         cmap='gray')
        ax[2].set_title('Transformed Greyscale Image', 
                         fontsize = f_size); 
    fourier_masker_ver(dark_image, 1)

    #Fourier Transform Horizontal Masked Image
    def fourier_masker_hor(image, i):
        f_size = 15
        dark_image_grey_fourier =    np.fft.fftshift(np.fft.fft2(rgb2gray(image)))
        dark_image_grey_fourier[235:240, :230] = i
        dark_image_grey_fourier[235:240,-230:] = i
        fig, ax = plt.subplots(1,3,figsize=(15,15))
        ax[0].imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
        ax[0].set_title('Masked Fourier', fontsize = f_size)
        ax[1].imshow(rgb2gray(image), cmap = 'gray')
        ax[1].set_title('Greyscale Image', fontsize = f_size);
        ax[2].imshow(abs(np.fft.ifft2(dark_image_grey_fourier)), 
                         cmap='gray')
        ax[2].set_title('Transformed Greyscale Image', 
                         fontsize = f_size);
    fourier_masker_hor(dark_image, 1)

    #Iterations of Masking Values
    def fourier_iterator(image, value_list):
        for i in value_list:
            fourier_masker_ver(image, i) 
    fourier_iterator(dark_image, [0.001, 1, 100])

    #Fourier Transformed Colored Image
    def fourier_transform_rgb(image):
        f_size = 25
        transformed_channels = []
        for i in range(3):
            rgb_fft = np.fft.fftshift(np.fft.fft2((image[:, :, i])))
            rgb_fft[:225, 235:237] = 1
            rgb_fft[-225:,235:237] = 1
            transformed_channels.append(abs(np.fft.ifft2(rgb_fft)))
        
        final_image = np.dstack([transformed_channels[0].astype(int), 
                                 transformed_channels[1].astype(int), 
                                 transformed_channels[2].astype(int)])
        
        fig, ax = plt.subplots(1, 2, figsize=(17,12))
        ax[0].imshow(image)
        ax[0].set_title('Original Image', fontsize = f_size)
        ax[0].set_axis_off()
        
        ax[1].imshow(final_image)
        ax[1].set_title('Transformed Image', fontsize = f_size)
        ax[1].set_axis_off()        
        fig.tight_layout()  
    
    fourier_transform_rgb(dark_image)
    
    
def Edge_Detection():
      
    # Opening the image (R prefixed to string
    # in order to deal with '\' in paths)
    image = Image.open(r"E:\sem 3\spyder practise\Project\Edge_Detection_Image.png")
      
    # Converting the image to grayscale, as edge detection 
    # requires input image to be of mode = Grayscale (L)
    image = image.convert("L")
      
    # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
    image = image.filter(ImageFilter.FIND_EDGES)
      
    # Saving the Image Under the name Edge_Detection_Output.png
    image.save(r"Edge_Detection_Output.png")


def Wavelet_Transforms():
    
    # Load image
    original = pywt.data.camera()

    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

# Main
while True: # This simulates a Do Loop
    choice = int(input(
        "MENU:\n   1. Apply a Gauss filter to an image.\n   2. Edge Detection using Pillow.\n   3. Morphological Operations.\n   4. Wavelet Transforms.\n   5. Application of Fourier Transformation.\n   6. Exit\nEnter the number corresponding to the menu to implement the choice: ")) # Menu Based Implementation

    if choice == 1:
        Gaussian_Filter() # Apply a Gauss filter to an image
    elif choice == 2:
        Edge_Detection() # Edge Detection using Pillow
    elif choice == 3:
        Morphological_Operations() # Morphological Operations
    elif choice == 4:
        Wavelet_Transforms() # Wavelet Transforms
    elif choice == 5:
        Fourier_Tranform() # Application of Fourier Transformation
    elif choice == 6: 
        break  # Exit loop
    else:
        print("Error: Invalid Input! Please try again.")
