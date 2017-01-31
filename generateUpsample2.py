import os, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

os.chdir('/home/sgholami/')
if not os.path.exists('trainsetSimple'):
    os.makedirs('trainsetSimple')
os.chdir('trainsetSimple')


def create_image(xmean, height, width, xmin, xmax, xeps, num_of_images = 50):
    #width = int(width)i
    #height = int(height)
    #num_of_images = int(num_of_images)
    temp = list()
    xsize = height * width
    current = time.strftime("%Y%m%d%H%M%S")


    for n in xrange(num_of_images):
        filename = 'img{}_{}.png'.format(current,n)

        x = np.random.randint(xmin,xmax,size=xsize)
        while abs(xmean - np.mean(x)) >= xeps:
            if xmean > np.mean(x):
                less = np.where(x < xmean)
                x[less[0][np.random.choice(range(0,len(less[0])))]] = np.random.randint(xmean,xmax,size=1)
            elif xmean < np.mean(x):
                more = np.where(x > xmean)
                x[more[0][np.random.choice(range(0,len(more[0])))]] = np.random.randint(xmin,xmean,size=1)

        imarray = x.reshape((height, width))
        imarray = imarray.repeat(14, axis=0).repeat(14, axis=1)
        print imarray
        image = Image.fromarray(imarray.astype('uint8')).convert('L')
        plt.imshow(imarray, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        #plt.savefig(dir2 + '/' + filename)
        #image.save('result_image.png')
        #plt.imshow(image, cmap='gray', interpolation='nearest')
        #plt.show()   

        image.save(dir2 + '/' + filename)
        temp.append(np.mean(x))

    return temp

start_time = time.time()
ranged = [0, 15, 45, 75, 105, 135, 165, 195, 225, 255]
#for i in ranged:
#for i in range(15, 285, 30):
for i in [225]:
    dir2 = '{}'.format(i)
    if not os.path.exists(dir2):
        os.makedirs(dir2)
    #std = min(i, (255-i))
    desiredMean = i
    width = 2
    height = 2
    minPixelBrightness = 0
    maxPixelBrightness = height * width * desiredMean
    if maxPixelBrightness > 256:
        maxPixelBrightness = 256
    if i==0:
        maxPixelBrightness = 1
    desiredPrecision = 0.01
    tmp2 = create_image(desiredMean, height, width, minPixelBrightness, maxPixelBrightness, desiredPrecision, num_of_images = 50000)
    # 65535 and 952 second for each level of brightness
    print tmp2

print("--- %s seconds ---" % (time.time() - start_time))