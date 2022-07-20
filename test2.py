import os
import time
import numpy as np
import warnings
import scipy
import SimpleITK as sitk
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.layers import concatenate
from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint
from keras import backend as K
#from keras.preprocessing.image import transform_matrix_offset_center
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages

K.set_image_data_format('channels_last')

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
smooth=1.

def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        #print("shape",target.get_shape()[2],refer.get_shape()[2])
        cw = (target.get_shape()[2] - refer.get_shape()[2])#.value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])#.value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

def get_unet(img_shape = None, first5=True):
        inputs = Input(shape = img_shape)
        concat_axis = -1

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

def Utrecht_preprocessing(FLAIR_image, T1_image):

    channel_num = 2
    print("utrecht",np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
    #print("133",(image_rows_Dataset/2-rows_standard/2),(image_rows_Dataset/2+rows_standard/2), (image_cols_Dataset/2-cols_standard/2),(image_cols_Dataset/2+cols_standard/2))
    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
    FLAIR_image = FLAIR_image[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_FLAIR = brain_mask_FLAIR[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    ###------Gaussion Normalization here
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
    T1_image = T1_image[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_T1 = brain_mask_T1[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])
    #---------------------------------------------------
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    T1_image  = T1_image[..., np.newaxis]
    imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis = 3)
    #print(np.shape(imgs_two_channels))
    return imgs_two_channels

def Utrecht_postprocessing(FLAIR_array, pred):
    start_slice = 6
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:,int((image_rows_Dataset-rows_standard)/2):int((image_rows_Dataset+rows_standard)/2),int((image_cols_Dataset-cols_standard)/2):int((image_cols_Dataset+cols_standard)/2)] = pred[:,:,:,0]
    
    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred

def GE3T_preprocessing(FLAIR_image, T1_image):
    print("GE3T",np.shape(FLAIR_image))
    channel_num = 2
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
  
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])

    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    FLAIR_image_suitable[:, :, int(cols_standard/2-image_cols_Dataset/2):int(cols_standard/2+image_cols_Dataset/2)] = FLAIR_image[:, start_cut:start_cut+rows_standard, :]
   
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
 
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])

    T1_image_suitable[...] = np.min(T1_image)
    T1_image_suitable[:, :, int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2)] = T1_image[:, start_cut:start_cut+rows_standard, :]
    #---------------------------------------------------
    FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable  = T1_image_suitable[..., np.newaxis]
    
    imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
    #print(np.shape(imgs_two_channels))
    return imgs_two_channels

def GE3T_postprocessing(FLAIR_array, pred):
    start_slice = 11
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_array)[0]
    image_rows_Dataset = np.shape(FLAIR_array)[1]
    image_cols_Dataset = np.shape(FLAIR_array)[2]
    original_pred = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:, start_cut:start_cut+rows_standard,:] = pred[:,:, int((rows_standard-image_cols_Dataset)/2):int((rows_standard+image_cols_Dataset)/2),0]

    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred

def test_leave_one_out(patient=0, path="FI1/",flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    if patient < 20: dir = 'raw/Utrecht/'
    elif patient < 40: dir = 'raw/Singapore/'
    else: dir = 'raw/GE3T/'
    dirs = os.listdir(dir)
    dirs.sort()
    dir += dirs[patient%20]
    FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
    T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_array = sitk.GetArrayFromImage(T1_image)
    if patient < 40: imgs_test = Utrecht_preprocessing(FLAIR_array, T1_array)
    else: imgs_test = GE3T_preprocessing(FLAIR_array, T1_array)
    if not flair: imgs_test = imgs_test[..., 1:2].copy()
    if not t1: imgs_test = imgs_test[..., 0:1].copy()
    img_shape = (rows_standard, cols_standard, flair+t1)
    model = get_unet(img_shape, first5)
    model_path = 'models/'+path
#if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
    model.load_weights(model_path + str(patient) + '.h5')
    pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.
    if patient < 40: original_pred = Utrecht_postprocessing(FLAIR_array, pred)
    else: original_pred = GE3T_postprocessing(FLAIR_array, pred)
    filename_resultImage = model_path + str(patient) + '.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
    filename_testImage = os.path.join(dir + '/wmh.nii.gz')
    testImage, resultImage = getImages(filename_testImage, filename_resultImage)
    dsc = getDSC(testImage, resultImage)
    avd = getAVD(testImage, resultImage) 
    #h95 = getHausdorff(testImage, resultImage)
    recall, f1 = getLesionDetection(testImage, resultImage)
    return dsc, avd, recall, f1

def main():
    result = np.ndarray((29,4), dtype = 'float32')
    dscs=[]
    avds=[]
    recalls=[]
    f1s=[]
    
    for patient in range(29):
        dsc, avd, recall, f1 = test_leave_one_out(patient,path="FI1/", first5=True, verbose=True)#
        dscs.append(dsc)
        avds.append((avd))
        recalls.append(recall)
        f1s.append(f1)
        print('Result of patient ' + str(patient))
        print('Dice',                dsc,       '(higher is better, max=1)')
       
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
        result[patient, 0] = dsc
        result[patient, 1] = avd
        result[patient, 2] = recall
        result[patient, 3] = f1
        #break
    

    np.save('resultsFI1.npy', result)



    result = np.ndarray((29,4), dtype = 'float32')
    dscs2=[]
    avds2=[]
    recalls2=[]
    f1s2=[]
    
    for patient in range(29):
        dsc, avd, recall, f1 = test_leave_one_out(patient,path="FI2/", first5=True, verbose=True)#
        dscs2.append(dsc)
        avds2.append((avd))
        recalls2.append(recall)
        f1s2.append(f1)
        print('Result of patient ' + str(patient))
        print('Dice',                dsc,       '(higher is better, max=1)')
       
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
        result[patient, 0] = dsc
        result[patient, 1] = avd
        result[patient, 2] = recall
        result[patient, 3] = f1
        #break
    

    np.save('resultsFI2.npy', result)


    result = np.ndarray((29,4), dtype = 'float32')
    dscs3=[]
    avds3=[]
    recalls3=[]
    f1s3=[]
    
    for patient in range(29):
        dsc, avd, recall, f1 = test_leave_one_out(patient,path="FI3/", first5=True, verbose=True)#
        dscs3.append(dsc)
        avds3.append((avd/100))
        recalls3.append(recall)
        f1s3.append(f1)
        print('Result of patient ' + str(patient))
        print('Dice',                dsc,       '(higher is better, max=1)')
       
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
        result[patient, 0] = dsc
        result[patient, 1] = avd
        result[patient, 2] = recall
        result[patient, 3] = f1
        #break
    

    np.save('resultsFI3.npy', result)


    result = np.ndarray((29,4), dtype = 'float32')
    dscs4=[]
    avds4=[]
    recalls4=[]
    f1s4=[]
    
    for patient in range(29):
        dsc, avd, recall, f1 = test_leave_one_out(patient,path="FI4/", first5=True, verbose=True)#
        dscs4.append(dsc)
        avds4.append((avd))
        recalls4.append(recall)
        f1s4.append(f1)
        print('Result of patient ' + str(patient))
        print('Dice',                dsc,       '(higher is better, max=1)')
       
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
        result[patient, 0] = dsc
        result[patient, 1] = avd
        result[patient, 2] = recall
        result[patient, 3] = f1
        #break
    

    np.save('resultsFI4.npy', result)    
    result = np.ndarray((29,4), dtype = 'float32')
    dscs5=[]
    avds5=[]
    recalls5=[]
    f1s5=[]
    
    for patient in range(29):
        dsc, avd, recall, f1 = test_leave_one_out(patient,path="FI5/", first5=True, verbose=True)#
        dscs5.append(dsc)
        avds5.append((avd))
        recalls5.append(recall)
        f1s5.append(f1)
        print('Result of patient ' + str(patient))
        print('Dice',                dsc,       '(higher is better, max=1)')
       
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
        result[patient, 0] = dsc
        result[patient, 1] = avd
        result[patient, 2] = recall
        result[patient, 3] = f1
        #break
    
    np.save('resultsFI5.npy', result)

    fig=plt.figure(figsize=(18,6),dpi=150)
    plt.title(" DICE ")
    plt.plot(dscs)
    plt.plot(dscs2)
    plt.plot(dscs3)
    plt.plot(dscs4)
    plt.plot(dscs5)
    #plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['DICE', 'DICE2','DICE3','DICE4','DICE5'], loc='upper left')
    #plt.show()
    model_pathpic ='scores'+"dice" + '.png'
    fig.savefig(model_pathpic)

    fig=plt.figure(figsize=(18,6),dpi=150)
    plt.title("AVD Score")
    plt.plot(avds)
    plt.plot(avds2)
    plt.plot(avds3)
    plt.plot(avds4)
    plt.plot(avds5)
    #plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['AVD', 'AVD','AVD','AVD','AVD'], loc='upper left')
    #plt.show()
    model_pathpic ='scores'+"AVD"+ '.png'
    fig.savefig(model_pathpic)


    fig=plt.figure(figsize=(18,6),dpi=150)
    plt.title("RECALL ")
    plt.plot(recalls)
    plt.plot(recalls2)
    plt.plot(recalls3)
    plt.plot(recalls4)
    plt.plot(recalls5)
    #plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['RECALL', 'RECALL2','RECALL3','RECALL4','RECALL5'], loc='upper left')
    #plt.show()
    model_pathpic ='models'+"FI3" + '.png'
    fig.savefig(model_pathpic)

    fig=plt.figure(figsize=(18,6),dpi=150)
    plt.title("F1 ")
    plt.plot(f1s)
    plt.plot(f1s2)
    plt.plot(f1s3)
    plt.plot(f1s4)
    plt.plot(f1s5)
    #plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['F1-1', 'F1-2','F1-3','F1-4','F1-5'], loc='upper left')
    #plt.show()
    model_pathpic ='models'+"FI4" + '.png'
    fig.savefig(model_pathpic)

    dice=[]
    dice.append(sum(dscs)/len(dscs))   
    dice.append(sum(dscs2)/len(dscs)) 
    dice.append(sum(dscs3)/len(dscs)) 
    dice.append(sum(dscs4)/len(dscs)) 
    dice.append(sum(dscs5)/len(dscs))  
    fig=plt.figure(figsize=(18,6),dpi=150)
    plt.title(" DICE Average ")
    plt.plot(dice)
    #plt.title('model accuracy')
    plt.ylabel('Average')
    plt.xlabel('epoch')
    #plt.legend(['DICE', 'DICE2','DICE3','DICE4','DICE5'], loc='upper left')
    #plt.show()
    model_pathpic ='scoresaverage'+"dice" + '.png'
    fig.savefig(model_pathpic)

    AVD=[]
    AVD.append(sum(avds)/len(avds))
    AVD.append(sum(avds2)/len(avds))
    AVD.append(sum(avds3)/len(avds))
    AVD.append(sum(avds4)/len(avds))
    AVD.append(sum(avds5)/len(avds))
    fig=plt.figure(figsize=(18,6),dpi=150)
    plt.title("AVD Average")
    plt.plot(AVD)
    #plt.title('model accuracy')
    plt.ylabel('Average')
    plt.xlabel('epoch')
    #plt.legend(['AVD', 'AVD','AVD','AVD','AVD'], loc='upper left')
    #plt.show()
    model_pathpic ='scoresaverage'+"AVD"+ '.png'
    fig.savefig(model_pathpic)

    rec=[]
    rec.append(sum(recalls)/len(recalls))
    rec.append(sum(recalls2)/len(recalls))
    rec.append(sum(recalls3)/len(recalls))
    rec.append(sum(recalls4)/len(recalls))
    rec.append(sum(recalls5)/len(recalls))
    fig=plt.figure(figsize=(18,6),dpi=150)
    plt.title("RECALL ")
    plt.plot(rec)
    
    #plt.title('model accuracy')
    plt.ylabel('Average')
    plt.xlabel('epoch')
    #plt.legend(['RECALL', 'RECALL2','RECALL3','RECALL4','RECALL5'], loc='upper left')
    #plt.show()
    model_pathpic ='modelsscores'+"recall" + '.png'
    fig.savefig(model_pathpic)
    FA=[]
    FA.append(sum(f1s)/len(f1s))
    FA.append(sum(f1s2)/len(f1s))
    FA.append(sum(f1s3)/len(f1s))
    FA.append(sum(f1s4)/len(f1s))
    FA.append(sum(f1s5)/len(f1s))
    fig=plt.figure(figsize=(18,6),dpi=150)
    plt.title("F1 ")
    plt.plot(FA)
  
    #plt.title('model accuracy')
    plt.ylabel('Average')
    plt.xlabel('epoch')
    #plt.legend(['F1-1', 'F1-2','F1-3','F1-4','F1-5'], loc='upper left')
    #plt.show()
    model_pathpic ='modelsaverage'+"F1" + '.png'
    fig.savefig(model_pathpic)

    print("DICE    " ,dice,"AVD    ",AVD,"RECALL  ",rec,"F1  ",FA)

if __name__=='__main__':
    main()
