# DyFraNet
### If you are using our dataset immatrix_2D.npy, you can simply run the python code to train the model by:
python3 main.py --batch_size 32 

### If you are using your own dataset, you might need to specify the number of frames for the input to train the model by:
python3 main.py --batch_size 32
                --numframe N

### To download our pre-trained model, please download and save it as './model/...' separately from the link below:

https://www.dropbox.com/s/9phk9osmzzpbh66/model.zip?dl=0

### and then run prediction.ipynb to play around with your own input!


Copyright (c) 2022 Markus J. Buehler and others

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
