from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import difflib
import SimpleITK as sitk
import scipy.spatial
from keras.models import Model
from keras.layers import Input, concatenate, Activation,Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages
#from keras.callbacks import ModelCheckpoint
from keras import backend as K
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# -define u-net architecture--------------------
smooth = 1.
def dice_coef_for_training(y_true, y_pred):
    print(np.shape(y_pred))
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    print(np.shape(y_pred))
    print(np.shape(y_true))
    return -dice_coef_for_training(y_true, y_pred)

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        print(target.get_shape()[2],refer.get_shape()[2])
        #cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        cw = target.get_shape()[2] - refer.get_shape()[2]
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        #ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        ch = target.get_shape()[1] - refer.get_shape()[1]
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu
def get_unet(img_shape = None):

        data_format = 'channels_first'
        
        inputs = Input(shape = img_shape)
        concat_axis = -1
        filters=5
        conv1 = conv_bn_relu(64, filters, inputs)
        conv1 = conv_bn_relu(64, filters, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = conv_bn_relu(96, 3, pool1)
        conv2 = conv_bn_relu(96, 3, conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = conv_bn_relu(128, 3, pool2)
        conv3 = conv_bn_relu(128, 3, conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = conv_bn_relu(256, 3, pool3)
        conv4 = conv_bn_relu(256, 4, conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = conv_bn_relu(512, 3, pool4)
        conv5 = conv_bn_relu(512, 3, conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = conv_bn_relu(256, 3, up6)
        conv6 = conv_bn_relu(256, 3, conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = conv_bn_relu(128, 3, up7)
        conv7 = conv_bn_relu(128, 3, conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = conv_bn_relu(96, 3, up8)
        conv8 = conv_bn_relu(96, 3, conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = conv_bn_relu(64, 3, up9)
        conv9 = conv_bn_relu(64, 3, conv9)

        ch, cw = get_crop_shape(inputs, conv9)
        conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(learning_rate=(2e-4)), loss=dice_coef_loss)
        
        return model

#--------------------------------------------------------------------------------------
def preprocessing(FLAIR_array, T1_array):
    
    brain_mask = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    brain_mask[FLAIR_array >=thresh] = 1
    brain_mask[FLAIR_array < thresh] = 0
    for iii in range(np.shape(FLAIR_array)[0]):
        brain_mask[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask[iii,:,:])  #fill the holes inside brain
    
    FLAIR_array -=np.mean(FLAIR_array[brain_mask == 1])      #Gaussion Normalization
    FLAIR_array /=np.std(FLAIR_array[brain_mask == 1])
    
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    FLAIR_array = FLAIR_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
    if two_modalities:
        T1_array -=np.mean(T1_array[brain_mask == 1])      #Gaussion Normalization
        T1_array /=np.std(T1_array[brain_mask == 1])
        T1_array = T1_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
        imgs_two_channels = np.concatenate((FLAIR_array[..., np.newaxis], T1_array[..., np.newaxis]), axis = 3)
        return imgs_two_channels
    else: 
        return FLAIR_array[..., np.newaxis]


def postprocessing(FLAIR_array, pred):
    start_slice = int(np.shape(FLAIR_array)[0]*per)
    num_o = np.shape(FLAIR_array)[1]  # original size
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    original_pred = np.zeros(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[:,int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard,int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard] = pred[:,:,:,0]
    original_pred[0: start_slice, ...] = 0
    original_pred[(num_o-start_slice):num_o, ...] = 0
    return original_pred

## some pr-edefined parameters 
rows_standard = 200  #the input size 
cols_standard = 200
thresh = 30   # threshold for getting the brain mask
per = 0.125
two_modalities = True  # set two modalities or single modality as the input
compute_metric = False # if you want to compute some evaluation metric between the segmentation result and the groundtruth 

inputDir = 'data_example/4'
outputDir = 'result'
if not os.path.exists(outputDir):
    os.mkdir(outputDir)




#Read data----------------------------------------------------------------------------
if two_modalities:
    img_shape=(rows_standard, cols_standard, 2)
    model_dir = 'pretrained_FLAIR_T1'
    FLAIR_image = sitk.ReadImage(os.path.join(inputDir, 'FLAIR.nii.gz'))
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_image = sitk.ReadImage(os.path.join(inputDir, 'T1.nii.gz'))
    T1_array = sitk.GetArrayFromImage(T1_image)
    imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array))  # data preprocessing 
else:
    img_shape=(rows_standard, cols_standard, 1)
    model_dir = 'pretrained_FLAIR_only'
    FLAIR_image = sitk.ReadImage(os.path.join(inputDir, 'FLAIR.nii.gz')) #data preprocessing 
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_array = []
    imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array)) 


#Load model---------------------------------------------
try:
    model = get_unet(img_shape) 
    model.load_weights(os.path.join(model_dir,'0.h5'))  # 3 ensemble models
    print('-'*30)
    print('Predicting masks on test data...') 
    pred_1 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '1.h5')) 
    pred_2 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '2.h5'))
    pred_3 = model.predict(imgs_test, batch_size=1, verbose=1)
    pred = (pred_1+pred_2+pred_3)/3
    pred[pred[...,0] > 0.45] = 1      #0.45 thresholding 
    pred[pred[...,0] <= 0.45] = 0

    original_pred = postprocessing(FLAIR_array, pred) # get the original size to match

    #Save data-------------------------------------------------------
    print("Output Saved")
    filename_resultImage = os.path.join(outputDir,'out_mask4.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )

    # 
    if compute_metric: 
        filename_testImage = os.path.join(inputDir + '/FLAIR.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
    #    h95 = getHausdorff(testImage, resultImage) # the calculation of H95 has some issues in python 3+. 
        recall, f1 = getLesionDetection(testImage, resultImage)
        print('Result of prediction:')
        print('Dice',                dsc,       ('higher is better, max=1'))
    #    print('HD',                  h95, 'mm',  '(lower is better, min=0)')
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
except Exception as e: print(e)



inputDir = 'data_example/6'
outputDir = 'result'
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

#Read data----------------------------------------------------------------------------
if two_modalities:
    img_shape=(rows_standard, cols_standard, 2)
    model_dir = 'pretrained_FLAIR_T1'
    FLAIR_image = sitk.ReadImage(os.path.join(inputDir, 'FLAIR.nii.gz'))
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_image = sitk.ReadImage(os.path.join(inputDir, 'T1.nii.gz'))
    T1_array = sitk.GetArrayFromImage(T1_image)
    imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array))  # data preprocessing 
else:
    img_shape=(rows_standard, cols_standard, 1)
    model_dir = 'pretrained_FLAIR_only'
    FLAIR_image = sitk.ReadImage(os.path.join(inputDir, 'FLAIR.nii.gz')) #data preprocessing 
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_array = []
    imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array)) 


#Load model---------------------------------------------
try:
    model = get_unet(img_shape) 
    model.load_weights(os.path.join(model_dir,'0.h5'))  # 3 ensemble models
    print('-'*30)
    print('Predicting masks on test data...') 
    pred_1 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '1.h5')) 
    pred_2 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '2.h5'))
    pred_3 = model.predict(imgs_test, batch_size=1, verbose=1)
    pred = (pred_1+pred_2+pred_3)/3
    pred[pred[...,0] > 0.45] = 1      #0.45 thresholding 
    pred[pred[...,0] <= 0.45] = 0

    original_pred = postprocessing(FLAIR_array, pred) # get the original size to match

    #Save data-------------------------------------------------------
    print("Output Saved")
    filename_resultImage = os.path.join(outputDir,'out_mask6.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )

    # 
    if compute_metric: 
        filename_testImage = os.path.join(inputDir + '/FLAIR.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
    #    h95 = getHausdorff(testImage, resultImage) # the calculation of H95 has some issues in python 3+. 
        recall, f1 = getLesionDetection(testImage, resultImage)
        print('Result of prediction:')
        print('Dice',                dsc,       ('higher is better, max=1'))
    #    print('HD',                  h95, 'mm',  '(lower is better, min=0)')
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
except Exception as e: print(e)



inputDir = 'data_example/8'
outputDir = 'result'
if not os.path.exists(outputDir):
    os.mkdir(outputDir)

#Read data----------------------------------------------------------------------------
if two_modalities:
    img_shape=(rows_standard, cols_standard, 2)
    model_dir = 'pretrained_FLAIR_T1'
    FLAIR_image = sitk.ReadImage(os.path.join(inputDir, 'FLAIR.nii.gz'))
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_image = sitk.ReadImage(os.path.join(inputDir, 'T1.nii.gz'))
    T1_array = sitk.GetArrayFromImage(T1_image)
    imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array))  # data preprocessing 
else:
    img_shape=(rows_standard, cols_standard, 1)
    model_dir = 'pretrained_FLAIR_only'
    FLAIR_image = sitk.ReadImage(os.path.join(inputDir, 'FLAIR.nii.gz')) #data preprocessing 
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_array = []
    imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array)) 


#Load model---------------------------------------------
try:
    model = get_unet(img_shape) 
    model.load_weights(os.path.join(model_dir,'0.h5'))  # 3 ensemble models
    print('-'*30)
    print('Predicting masks on test data...') 
    pred_1 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '1.h5')) 
    pred_2 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '2.h5'))
    pred_3 = model.predict(imgs_test, batch_size=1, verbose=1)
    pred = (pred_1+pred_2+pred_3)/3
    pred[pred[...,0] > 0.45] = 1      #0.45 thresholding 
    pred[pred[...,0] <= 0.45] = 0

    original_pred = postprocessing(FLAIR_array, pred) # get the original size to match

    #Save data-------------------------------------------------------
    print("Output Saved")
    filename_resultImage = os.path.join(outputDir,'out_mask8.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )

    # 
    if compute_metric: 
        filename_testImage = os.path.join(inputDir + '/FLAIR.nii.gz')
        testImage, resultImage = getImages(filename_testImage, filename_resultImage)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage) 
    #    h95 = getHausdorff(testImage, resultImage) # the calculation of H95 has some issues in python 3+. 
        recall, f1 = getLesionDetection(testImage, resultImage)
        print('Result of prediction:')
        print('Dice',                dsc,       ('higher is better, max=1'))
    #    print('HD',                  h95, 'mm',  '(lower is better, min=0)')
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
except Exception as e: print(e)

















