# DyFraNet

![image](https://user-images.githubusercontent.com/101393859/202016081-178b33cc-6034-41b0-b656-83b56ae0ce83.png)

Reference: Yu-Chuan Hsu, Markus J. Buehler, DyFraNet: Forecasting and Backcasting Dynamic Fracture Mechanics in Space and Time Using a 2D-to-3D Deep Neural Network, in submission 

#### If you are using our dataset $immatrix\\_2D.npy$, you can simply run the python code to train the model by:
python3 main.py --batch_size 32 

#### If you are using your own dataset, you might need to specify the number of frames, $N$, for the input to train the model by:
python3 main.py --batch_size 32
                --numframe N

#### To download our pre-trained model, please download and unzip it to the currnet folder from the link below:

https://www.dropbox.com/s/9phk9osmzzpbh66/model.zip?dl=0

#### and then run $prediction.ipynb$ to explore the model with your own input.
