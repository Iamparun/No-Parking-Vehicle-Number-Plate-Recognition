import numpy as np
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from object_detection.utils import ops as utils_ops
import cv2
import datetime
from PIL import Image
import pytesseract
from time import sleep
import os
import smtplib


##from six.moves import urllib
##import imghdr

b=0
d=0
e=0

import smtplib
from email.message import EmailMessage
import imghdr
from time import sleep
email_add = 'techgalwin.tamil26@gmail.com'
email_pass = "iyjmxuyrtlvnqysu"
msg = EmailMessage()
msg['Subject'] = "NO PARKING AREA"
msg['From'] = "techgalwin.tamil26@gmail.com"
msg['To'] = "santhiarun2001@gmail.com"
msg.set_content("In the no parking zone in parking is prohibited FINE COLLECTED RS.1000")
def email():
    with open('capture.jpg','rb')as f:
        file_data = f.read()
        file_type = imghdr.what(f.name)
        file_name = f.name
    msg.add_attachment(file_data, maintype = 'image', subtype = file_type, filename = file_name) 
    with smtplib.SMTP_SSL('smtp.gmail.com',465)as smtp:
        smtp.login(email_add,email_pass)
        smtp.send_message(msg)











def Number_Plate_Recogniztion():
  img =cv2.imread("capture.jpg")
  text = pytesseract.image_to_string(img, config ='')
  #text = input("vehicle no : ")
  print(len(text))
  print(text)
  sleep(1)
  global b
  if len(text)>=12:
      file = open("result.txt","w")
      file.write(text)
      file.close()
      sleep(1)
      print("Text file Write")
      sleep(1)
      print("Text file Read")
      file = open("result.txt","r")
      a=file.read()
      file.close()
      b=a[0]+a[1]+a[2]+a[3]+a[4]+a[5]+a[6]+a[7]+a[8]+a[9]
      print(b)
      print("NUMBER PLATE RECOGNIZED......")
      sleep(1)
      


#how is it



# pytesseract

pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/labelmap.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
        
    return output_dict

    
A = 0
import cv2
cap = cv2.VideoCapture(0)
#url='http://192.168.1.26:8080/shot.jpg'
try:
    with detection_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)

                while True:
                
                    ret, image_np = cap.read()
                    '''imgPath = urllib.request.urlopen(url)
                    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
                    image_np = cv2.imdecode(imgNp, -1)'''
                    frame = cv2.resize(image_np,(800,600))
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, detection_graph)
                    if output_dict['detection_classes'][0] == 2 and  output_dict['detection_scores'][0] > 0.90 :
                      print('TRACTOR')
                      A=1
                      # Visualization of the results of a detection.
                      vis_util.visualize_boxes_and_labels_on_image_array(
                          image_np,
                          output_dict['detection_boxes'],
                          output_dict['detection_classes'],
                          output_dict['detection_scores'],
                          category_index,
                          instance_masks=output_dict.get('detection_masks'),
                          use_normalized_coordinates=True,
                          line_thickness=8)
                      cv2.imwrite("capture.jpg",image_np)
                          
                    if output_dict['detection_classes'][0] == 1 and  output_dict['detection_scores'][0] > 0.90 :
                      print('NO PARKING')
                      cv2.imwrite("capture.jpg",image_np)
                      sleep(2)
                      print("email sending in process...")
                      email()
                      print("email sending successfully....")
                      Number_Plate_Recogniztion()
                      sleep(2)
                      #sms()
                      sleep(2)
                      A=1
                      # Visualization of the results of a detection.
                      vis_util.visualize_boxes_and_labels_on_image_array(
                          image_np,
                          output_dict['detection_boxes'],
                          output_dict['detection_classes'],
                          output_dict['detection_scores'],
                          category_index,
                          instance_masks=output_dict.get('detection_masks'),
                          use_normalized_coordinates=True,
                          line_thickness=8)
                      cv2.imwrite("capture.jpg",image_np)
                    if A == 1:
                      print("Number Plate Recognize")
                      Number_Plate_Recogniztion()#function calling
                    cv2.imshow('Frame', image_np)
                    cv2.imwrite("capture.jpg",image_np)
                    if cv2.waitKey(1) == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
except Exception as e:
    print(e)
    cap.release()
