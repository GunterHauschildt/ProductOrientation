# ProductOrientation

Results  
-------  
https://github.com/user-attachments/assets/17b83546-8035-463e-9e69-8c63e64e77b5

Discussion  
----------  
Product orientation is often necessary in product handling - whether to instruct a robot to straighten or instruct a robot on how to grab the product.  

Orientation is angular. Angle operations in a neural network are hard to handle.  
1) There is a discontinuity at 360deg. Imagine for example a naive angle averaging routine around 0deg / 360deg: (1deg + 359deg) / 2 != 0.  
Instead the answer is to work on cosine and sine values.  
2) tan = sin/cos is noisey with values near 90deg. The divide by 'cos(90) = 0' causes small errors to be amplified.  
Instead the answer is work near 45degrees where cosine and sine are more linear.  
A 45deg phase shifted cosine / sine pair is added to the raw cosine / sine pair.  
As such our NN has 4 outputs: sin(theta + 0), cos(theta + 0), sin(theta + 45), cos(theta + 45)
  
Custom Loss  
-----------  
Part of this exercise is custom loss to work on only the linear-most cosine/sine pair.    
We can just sum the absolute values of the sine/cosine pairs to find which are closest to 45deg.  
(.707 + .707) > (1.00 + 0).

Regresion CNN  
-------------  
Part of this exercise is a regression CNN - instead of the more common 'classification' network.  
Pretrained networks were tried but didn't do as well as a custom CNN with 'leaky_relu' on all layers.  
Cosine/sine values are mapped from -1 to +1 to 0 to +1 to match the linear half of 'relu'.  
Notes on optimizer: Adam NaN's. SGD is used instead. TBD!

DataSet  
-------  
MVTec's Anomaly Detect's 'Hazelnut' dataset is use. Finding the orientation of a round object thought to challenge this algorithm.  



   


 
