#! /usr/bin/env python
"""
Train a YOLO_v2 model to Pascal VOC2007+2012 dataset.
"""
import argparse
import colorsys
from datetime import datetime
import h5py
import imghdr
import io
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from keras import backend as K
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Lambda
from keras.models import Model
from keras.models import load_model
from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes
import sys
import tqdm
IOU_THRESHOLD = 0.5
Precision_acc = 0
Average_precision = 0
cls_precision_acc = np.zeros([4])
cls_box_number = np.zeros([4])

def calculate_IoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]
 
    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]
 
    carea = (cx2 - cx1) * (cy2 - cy1) #C的面积
    garea = (gx2 - gx1) * (gy2 - gy1) #G的面积
 
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h #C∩G的面积
 
    iou = area / (carea + garea - area)
 
    return iou


YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))
detectors_mask_shape = (13, 13, 5, 1)
matching_boxes_shape = (13, 13, 5, 5)
batch_size = 8

argparser = argparse.ArgumentParser(
    description='Train YOLO_v2 model to overfit on a single image.')

argparser.add_argument(
    '-d',
    '--data_path',
    help='path to HDF5 file containing pascal voc dataset',
    #default='~/datasets/PascalVOC/VOCdevkit/pascal_voc_07_12.hdf5')
    default='/var/tmp/YOLO/YAD2K-master/detectiondata/viva.hdf5')
argparser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to images/out',
    default='images/out')
argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/yolo_anchors.txt')

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default='model_data/viva_classes.txt')
argparser.add_argument(
    '-t',
    '--training',
    action='store_true',
    help='training the model')

def _main(args):
    voc_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    is_training = args.training
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
    else:
        anchors = YOLO_ANCHORS

    voc = h5py.File(voc_path, 'r')
    total_img_number = voc['train/images'].shape[0]
    


    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)
    roi_input = Input(shape=(None, 4))
    
    num_rois = 4
    pooling_regions = 13
      
    # Create model body.
    model_body = yolo_body(image_input, roi_input, len(anchors), len(class_names))
    model_body = Model([image_input, roi_input], model_body.output)
    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])
    model = Model(
        [image_input, roi_input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,
                                allow_growth=True)
    config=tf.ConfigProto(gpu_options=gpu_options)
    set_session(tf.Session(config=config))
    iteration = 10
    num_steps = 1
    saved_file_name = 'iter40_batch8_weights.h5'
    max_num_box = [x.shape[0] for x in voc['train/boxes']]
    
    '''
    for x in voc['train/images']:
      print(x)
    '''
    #print("print imgs done")
    max_num_box = max(max_num_box) // 5
    #print("max_num_box = {}".format(max_num_box))
    #print("total_img_number = {}".format(total_img_number)) #10728
    batch_numbers = total_img_number//batch_size            #10728/24    
    new_saved_file_name = 'new_iter50_batch8_weights.h5'
    model.load_weights(saved_file_name)
    #print("is_training = {}".format(is_training))
    if is_training:
        for ite in range(iteration):
            random.seed(ite)
            #print("{} : iteration{} start".format(datetime.now(),ite))
            choose_list = list(range(0, total_img_number))
            for batch_number in range(batch_numbers): #660
                if (batch_number%30==0):
                    model.save_weights(new_saved_file_name)
                    #print("{} : processing No.{} batch".format(datetime.now(),batch_number))
                image_data_batch = []
                roi_inputs_batch = []
                boxes_batch = []
                detectors_mask_batch = []
                matching_true_boxes_batch = []
                batch_start = batch_number*batch_size
                for i in range(batch_size):
                    #random_index = random.choice(choose_list)
                    #choose_list.remove(random_index)
                    image = PIL.Image.open(io.BytesIO(voc['train/images'][i]))
                    orig_size = np.array([image.width, image.height])
                    orig_size = np.expand_dims(orig_size, axis=0)
                    # Image preprocessing.
                    image = image.resize((416, 416), PIL.Image.BICUBIC)
                    image_data = np.array(image, dtype=np.float)
                    image_data /= 255.
                    # Box preprocessing.
                    # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
                    boxes = voc['train/boxes'][i]
                    boxes = boxes.reshape((-1, 5))
                    # Get extents as y_min, x_min, y_max, x_max, class for comparision with
                    # model output.
                    boxes_extents = boxes[:, [2, 1, 4, 3, 0]]

                    # Get box parameters as x_center, y_center, box_width, box_height, class.
                    boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
                    boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
                    boxes_xy = boxes_xy / orig_size
                    boxes_wh = boxes_wh / orig_size
                    boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)
                    detectors_mask, matching_true_boxes = preprocess_true_boxes(boxes, anchors, [416, 416])
                    boxes = np.pad(boxes, ((0,max_num_box-boxes.shape[0]),(0,0)), 'constant')
                    
                        
                    # Random ROI for all images
                    roi_inputs = np.zeros((num_rois, 4))
                    for i in range(num_rois):
                        x = np.random.randint(0, pooling_regions-1, size=1)
                        y = np.random.randint(0, pooling_regions-1, size=1)
                        w = np.random.randint(1, pooling_regions-x, size=1)
                        h = np.random.randint(1, pooling_regions-y, size=1)
                        roi_inputs[i,:] = np.array([x,y,w,h]).flatten()
                    
                    image_data_batch.append(image_data.tolist())
                    roi_inputs_batch.append(roi_inputs.tolist())
                    boxes_batch.append(boxes.tolist())
                    detectors_mask_batch.append(detectors_mask.tolist())
                    matching_true_boxes_batch.append(matching_true_boxes.tolist())


                #print(matching_true_boxes.shape)  
                image_data_batch = np.array(image_data_batch)
                roi_inputs_batch = np.array(roi_inputs_batch)
                boxes_batch = np.array(boxes_batch)
                detectors_mask_batch = np.array(detectors_mask_batch)
                matching_true_boxes_batch = np.array(matching_true_boxes_batch)

                model.fit([image_data_batch, roi_inputs_batch, boxes_batch, detectors_mask_batch, matching_true_boxes_batch],
                          np.zeros(len(image_data_batch)),
                          batch_size=batch_size,
                          epochs=num_steps,
                          verbose=0)

            #print("{} : iteration{} end".format(datetime.now(),ite))
            model.save_weights(new_saved_file_name)
            #print("weight saved as {}".format(new_saved_file_name))
       # print("{} : all done".format(datetime.now()))
    
    """
    image = PIL.Image.open(io.BytesIO(voc['train/images'][10]))
    orig_size = np.array([image.width, image.height])
    orig_size = np.expand_dims(orig_size, axis=0)
    # Image preprocessing.
    image = image.resize((416, 416), PIL.Image.BICUBIC)
    image_data = np.array(image, dtype=np.float)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis=0)
    
    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    print("yolo_outputs) = {}".format(yolo_outputs))
    input_image_shape = K.placeholder(shape=(2, ))
    
    out_boxes, out_scores, out_classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=.001, iou_threshold=.001)
 

    print("out_boxes : {}".format(type(out_boxes)))
    print("out_scores : {}".format(type(out_scores)))
    print("out_classes : {}".format(type(out_classes)))
    #visualization
    out_boxes_temp = out_boxes
    out_scores_temp = out_scores
    out_classes_temp = out_classes
    bbox_pred = []
    bbox_c = []
    bbox_preds = []
    bbox_cls = []  
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))    
    try:
        max_score = max(out_scores)
        max_index = [i for i,j in enumerate(out_scores) if j==max_score]
        choosed_boxes = 0
        #print(len(out_scores))
        max_boxes = 1
        bbox_cls.append(np.array(bbox_c))
        bbox_preds.append(np.array(bbox_pred))
    except:
        xy_out_boxes = [0,0,10,10]
        bbox_pred.append(xy_out_boxes)
        bbox_c.append(0)
        bbox_cls.append(np.array(bbox_c))
        bbox_preds.append(np.array(bbox_pred))
        #print("fail {}".format(img_num))
        print("fail")

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    if True:
        out_boxes = out_boxes_temp
        out_scores = out_scores_temp
        out_classes = out_classes_temp
        #print("Number:{} processed ,Found {} boxes".format(img_num,len(out_boxes)))
        #draw the bounding box on image
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        image.save(os.path.join(output_path, image_file), quality=90)
    """
    for idx in range(0,100,100):
      image = PIL.Image.open(io.BytesIO(voc['train/images'][idx]))
      orig_size = np.array([image.width, image.height])
      orig_size = np.expand_dims(orig_size, axis=0)
      # Image preprocessing.
      image = image.resize((416, 416), PIL.Image.BICUBIC)
      image_data = np.array(image, dtype=np.float)
      image_data /= 255.
      image_data = np.expand_dims(image_data, axis=0)
      
      roi_inputs = np.zeros((num_rois, 4))
      for i in range(num_rois):
          x = np.random.randint(0, pooling_regions-1, size=1)
          y = np.random.randint(0, pooling_regions-1, size=1)
          w = np.random.randint(1, pooling_regions-x, size=1)
          h = np.random.randint(1, pooling_regions-y, size=1)
          roi_inputs[i,:] = np.array([x,y,w,h]).flatten()
      roi_inputs = np.expand_dims(roi_inputs, axis=0)
      
      # Create output variables for prediction.
      yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
      #print("yolo_outputs) = {}".format(yolo_outputs))
      input_image_shape = K.placeholder(shape=(2, ))
      boxes, scores, classes = yolo_eval(
	  yolo_outputs, input_image_shape, score_threshold=.005, iou_threshold=.001)
      
      #print(type([image.size[1], image.size[0]]))
      #print(type(K.learning_phase()))
      #print(K.learning_phase())
      # Run prediction on overfit image.
      #import pdb
      #pdb.set_trace() #中斷點
      sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
      out_boxes, out_scores, out_classes = sess.run(
	  [boxes, scores, classes],
	  feed_dict={
	      model_body.input[0]: image_data,
	      model_body.input[1]: roi_inputs,
	      input_image_shape: [image.size[1], image.size[0]],
	      K.learning_phase(): 0
	  })
      #print('Found {} boxes for image.'.format(len(out_boxes)))
      #print(out_boxes)

      # Plot image with predicted boxes.
      image_with_boxes = draw_boxes(image_data[0], out_boxes, out_classes,
				    class_names, out_scores)
      #plt.imshow(image_with_boxes, interpolation='nearest')
      #plt.show()
    
    
    
    # Run prediction on overfit image.
    
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    
    ############
    
    
    #For competition2
    
    #print("------------")
    
    num_test_img=voc['test/images'].shape[0]
    #print(num_test_img)
    
    #oc_path = os.path.expanduser("~/datasets/VOCdevkit/VOC2012/JPEGImages")
    
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)      # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)       # Reset seed to default.
    
    
    output_path = os.path.expanduser(args.output_path)
    #voc_path = os.path.expanduser("~/datasets/VOCdevkit/VOC2012/JPEGImages")
    if not os.path.exists(output_path):
        #print('Creating output path {}'.format(output_path))
        os.mkdir(output_path)
    
    
    #print("-------------")
    '''
    
    #for img_num,image_file in enumerate(os.listdir(test_path)):
    '''
    '''
    try:
        image_type = imghdr.what(os.path.join(test_path, image_file))
        if not image_type:
            continue
    except IsADirectoryError:
        continue
    '''
    
    model_image_size = model.layers[0].input_shape[1:3]

    score_threshold = 0.05
    for img_num in range(num_test_img):
      
        image_file = str(img_num)+'.jpg'
        #print(image_file)
        image = PIL.Image.open(io.BytesIO(voc['test/images'][img_num]))
        #image = Image.open(os.path.join(test_path, image_file))
        if True:  # TODO: When resizing we can use minibatch input.
            resized_image = image.resize(
                tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            #print(image_data.shape)

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input[0]: image_data,
                model_body.input[1]: roi_inputs,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
    

        
        
        out_boxes_temp = out_boxes
        out_scores_temp = out_scores
        out_classes_temp = out_classes
      
        
        try:
            max_score = max(out_scores)
            max_index = [i for i,j in enumerate(out_scores) if j==max_score]
            choosed_boxes = 0
            #print(len(out_scores))
            max_boxes = 4
        except:
            xy_out_boxes = [0,0,10,10]
            #print("fail {}".format(img_num))
            
        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        
        ''''''
        out_boxes = out_boxes_temp
        out_scores = out_scores_temp
        out_classes = out_classes_temp
        out_scores_order = out_scores.argsort()[:][::-1]

        exits=[0,0,0,0]
        bbox_pred=[]
        bbox_c=[]
        bbscore=[]
        #draw the bounding box on image
        '''
        for index in range(len(out_scores)):        
            if (choosed_boxes<max_boxes) :                
                #max_score = max(out_scores)
                #max_index = [i for i,j in enumerate(out_scores) if j==max_score]
                max_index = out_scores_order[index]
                box = out_boxes[max_index]
                top, left, bottom, right = box
                xy_out_boxes = [left,top,right,bottom]
                bbox_pred.append(xy_out_boxes)
                bbox_c.append(out_classes[max_index])
                choosed_boxes +=1
        '''
        
        
        class_set=set(out_classes)
        for index in range(len(out_classes)):
          max_index = out_scores_order[index]
          if exits[out_classes[max_index]]==0 :
            exits[out_classes[max_index]]=1
            box = out_boxes[max_index]
            top, left, bottom, right = box
            xy_out_boxes = [left,top,right,bottom]
            bbox_pred.append(xy_out_boxes)
            bbox_c.append(out_classes[max_index])
            bbscore.append(out_scores[max_index])
            choosed_boxes +=1
            if choosed_boxes==len(class_set):
              break
        
        
        #use all pred boxes
        '''
        print(out_classes)
        class_set=set(out_classes)
        for index in range(len(out_classes)):
          max_index = out_scores_order[index]
          #if exits[out_classes[max_index]]==0 :
          exits[out_classes[max_index]]=1
          box = out_boxes[max_index]
          top, left, bottom, right = box
          xy_out_boxes = [left,top,right,bottom]
          bbox_pred.append(xy_out_boxes)
          bbox_c.append(out_classes[max_index])
          bbscore.append(out_scores[max_index])
          choosed_boxes +=1
           # if choosed_boxes==len(class_set):
            #  break
        '''
        # bbox_pred, bbox_c, bbscore
        #-------------- calculate IoU ------------------------------
        IoU =[]
        AP = 0
        hit = 0
        reshaped_gt = voc['test/boxes'][img_num].reshape(-1,5)
        gt_box_number = reshaped_gt.shape[0]
        #sort_bbox_c = np.argsort(bbox_c, axis=0)
        
        # for each gt box, check if it's hit or not
        for idx_gt in range(gt_box_number):
            gt_cls = reshaped_gt[idx_gt][0]
            cls_box_number[gt_cls] +=1
            # for each pred boxes
            for idx_pred,pred_cls in enumerate(class_set):
                #print("pred_cls={}  idx_gt={}".format(pred_cls,idx_gt))
                if pred_cls == gt_cls:
                    IoUtemp = calculate_IoU(bbox_pred[idx_pred], reshaped_gt[idx_gt][1:])
                    IoU.append(IoUtemp)
                    #print("IoUtemp = {}".format(IoUtemp))
                    if IoUtemp>= IOU_THRESHOLD:
                        cls_precision_acc[gt_cls] += 1 
        #print("Number:{} processed".format(img_num))
        if img_num%50==0:
            #print("Number:{} processed ,Found {} boxes".format(img_num,len(out_boxes)))
            #print(out_classes,out_scores,out_boxes)
            for i in range(len(bbox_pred)):
                predicted_class = class_names[i]
                box = bbox_pred[i]
                score = bbscore[i]
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                #print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for j in range(thickness):
                    draw.rectangle(
                        [left + j, top + j, right - j, bottom - j],
                        outline=colors[i])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[i])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
            image.save(os.path.join(output_path, image_file), quality=90)
        #end for
    Average_precision = 0
    for cls in range(4):
        clss_precision = cls_precision_acc[cls]/cls_box_number[cls]
        Average_precision += clss_precision
        #print("Precision of Class {} = {}".format(cls,clss_precision))
    Average_precision = Average_precision/4
    #print("Average precision = {}".format(Average_precision))
    sess.close()
    

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)