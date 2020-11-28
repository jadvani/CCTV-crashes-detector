# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:33:58 2020

@author: Javier
"""
acc = 0
no_acc = 0
total = len(onlyfiles)
for file in onlyfiles:
    img = image.load_img('F:\\TFM_datasets\\car-crashes-detector\\dataset\\resize\\test\\2\\'+file,False,target_size=(75,75))
#img = image.load_img('dataset/test/2/1978.jpg',False,target_size=(28,28))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    result = model.predict(x)
    if result[0][0] >= 0.5:
        acc = acc+1
    else:
        no_acc = no_acc+1

#%%
from math import sqrt
total_p = 360
tp = 341
fp = total_p - tp
total_n = 375

tn = 359
fn = total_p - tn
precision = tp / (tp+fp)
recall = tp / (tp+fn)
f1 = 2*(precision*recall)/(precision+recall)
mcc=((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
accuracy = (tp + tn) / (total_p+total_n)
print("mcc="+str(mcc))
print('accuracy: '+str(accuracy)+', precision: '+str(precision)+", recall: "+str(recall)+", f1:"+str(f1))