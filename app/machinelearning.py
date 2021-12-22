
import numpy as np
import cv2
import os
import pickle
import sklearn
from django.conf import settings

STATIC_DIR = settings.STATIC_DIR

face_detection_model = os.path.join(STATIC_DIR,'./models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
face_detection_proto=os.path.join(STATIC_DIR,'./models/deploy.prototxt.txt')
face_descriptor=os.path.join(STATIC_DIR,'./models/openface.nn4.small2.v1.t7')


# face detection
face_detector_model=cv2.dnn.readNetFromCaffe(face_detection_proto,face_detection_model)

#feature extraction
face_feature_model=cv2.dnn.readNetFromTorch(face_descriptor)

#face recognition
face_recognition_model= pickle.load(open(os.path.join(STATIC_DIR,'./models/machine_learning_face_person.pkl'),mode='rb'))

#emotion recognition
emotion_recognition_model=pickle.load(open(os.path.join(STATIC_DIR,'./models/emotion_model_new.pkl'),mode='rb'))


def pipeline_model(path):
  img=cv2.imread(path)
  image=img.copy()
  h,w=img.shape[:2]

  img_blob = cv2.dnn.blobFromImage(img,1,(300,300),(104,177,123),False,False)

  face_detector_model.setInput(img_blob)
  detections=face_detector_model.forward()
  #results
  ml_results = dict(face_detect_score=[],
                    face_name=[],
                    face_name_score=[],
                    emotion_name=[],
                    emotion_name_score=[],
                    count=[])
  count=1
  if len(detections)>0:
    for i, confidence in enumerate(detections[0,0,:,2]):
      if confidence>0.5:
        box=detections[0,0,i,3:7]
        box=box*np.array([w,h,w,h])
        startx,starty,endx,endy=box.astype(int)

        cv2.rectangle(image,(startx,starty),(endx,endy),(0,255,0),2)

        #feature extraction
        face_roi=img[starty:endy,startx:endx]
        face_blob=cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),True,True)
        face_feature_model.setInput(face_blob)
        vectors=face_feature_model.forward()

        #recognition and emotion
        face_name=face_recognition_model.predict(vectors)[0]
        face_score=face_recognition_model.predict_proba(vectors).max()

        text_face=f'{face_name} : {int(100*(face_score))}'

        emotion_name=emotion_recognition_model.predict(vectors)[0]
        emotion_score=emotion_recognition_model.predict_proba(vectors).max()
        
        text_emotion=f'{emotion_name} : {int(100*(emotion_score))}'
        cv2.putText(image,text_face,(startx,starty),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)
        cv2.putText(image,text_emotion,(startx,endy),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),3)

        cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'mloutput/process.jpg'),image)
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT,'mloutput/face_roi_{}.jpg'.format(count)),face_roi)

        ml_results['count'].append(count)
        ml_results['face_detect_score'].append(confidence*100)
        ml_results['face_name'].append(face_name)
        ml_results['face_name_score'].append(face_score*100)
        ml_results['emotion_name'].append(emotion_name)
        ml_results['emotion_name_score'] .append ((float)(emotion_score)*100)

        count+=1
  return ml_results
