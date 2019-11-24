#!/usr/bin/env python
# coding: utf-8

# In[6]:


import ocr_engine as engine
import cv2

#load the classifier
clf=engine.load('digit_classifier_knn.model')

name='nbrs.jpg'
img=cv2.imread(name)
digits=engine.perform_ocr(img,clf)

#make up the number from the individual digits
#and compute its suare
number=int(''.join(digits))
sqr=number**2

#display the information
forn=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,"sqr({})={}".format(number,sqr),(0,40),font,1,(0,255,255),2)
cv2.imshow("digits",img)
cv2.waitKey(0)

