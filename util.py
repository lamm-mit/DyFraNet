import numpy as np

def acc_crack(img_seq, r=0.8):
    length_ = [0]
    overall_img_temp = np.copy(img_seq[0])
    overall_img = [np.copy(overall_img_temp)]
    for l in range(1, img_seq.shape[0]):
        length_temp = 0
        for i in range(img_seq.shape[1]):
            for j in range(img_seq.shape[2]):
                  
                if img_seq[l][i,j] < r:
                    overall_img_temp[i,j] = 0
                if overall_img_temp[i,j] < r:
                    length_temp+=1
                    
        length_.append(length_temp)
        overall_img.append(np.copy(overall_img_temp))
        
    return np.array(overall_img), length_

def vabs(v):
    return (v[0]**2+v[1]**2)**0.5
def vdot(v1,v2):
    return (v1[0]*v2[0]+v1[1]*v2[1])
def sin_angle(v1,v2,p=np.pi/3*2, _print=False):
    angle = (p-np.arccos(vdot(v1,v2)/vabs(v1)/vabs(v2))+1e-15)%p
    if _print:
        print(str(angle*180/np.pi))
    return angle


