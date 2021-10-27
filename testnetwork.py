from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning) 

key = {
    "1" : "a",
    "2" : "b",
    "3" : "c",
    "4" : "d",
    "5" : "e",
    "6" : "f",
    "7" : "g",
    "8" : "h",
    "9" : "i",
    "10" : "j",
    "11" : "k",
    "12" : "l",
    "13" : "m",
    "14" : "n",
    "15" : "o",
    "16" : "p",
    "17" : "q",
    "18" : "r",
    "19" : "s",
    "20" : "t",
    "21" : "u",
    "22" : "v",
    "23" : "w",
    "24" : "x",
    "25" : "y",
    "26" : "z"
}

def processImage(img):
    arr = np.array(img)
    r,g,b = np.split(arr,3,axis=2)
    r=r.reshape(-1)
    g=r.reshape(-1)
    b=r.reshape(-1)
    bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2], zip(r,g,b)))
    bitmap = np.array(bitmap).reshape([arr.shape[0], arr.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float),255)
    im = Image.fromarray(bitmap.astype(np.uint8))
    im.save('result.jpg')
    
    rows = len(bitmap)
    cols = len(bitmap[0])

    for i in range(rows):
        for j in range(cols):
            bitmap[i][j] = (255 - bitmap[i][j]) / 255

    bitmap = np.array(bitmap.reshape(-1))

    return bitmap

def newLayer(newLength, oldLayer):
    weights = np.random.normal(loc=0, scale=5.0, size=(newLength, len(oldLayer)))

    biases = np.random.randint(low=-10, high=10, size=(newLength,))
  
    newLayer = weights@oldLayer
    newLayer = np.add(newLayer, biases)
    for i in range(len(newLayer)):
        newLayer[i] = round(newLayer[i], 10)
        newLayer[i] = sigmoid(newLayer[i])

    return newLayer, weights, biases

def cost(actual, result):
    total = 0;

    for i in range(len(result)):
        if i == actual - 1:
            total += (result[i] - 1)**2
        else:
            total += (result[i] - 1)**2

    return total

def sigmoid(x):
    return round(1/(1+(np.e**(-x))), 10)

if __name__ == "__main__":
    img = Image.open("img.jpg")
    print("Processing image... ")
    bitmap = processImage(img)
    print("Image processed")
    print("Creating layer 1... ")
    layer1, weights1, biases1 = newLayer(32, bitmap)
    print("Layer 1 complete")
    print("Creating layer 2... ")
    layer2, weights2, biases2 = newLayer(32, layer1)
    print("Layer 2 complete")
    print("Creating layer 3... ")
    layer3, weights3, biases3 = newLayer(26, layer2)
    print("Layer 3 complete")

    greatestNum = layer3[0]
    greatestIndex = 0

    for i in range(len(layer3)):
        if layer3[i] > greatestNum:
            greatestNum = layer3[i]
            greatestIndex = i
  
    cost = cost(5, layer3)
    print("Result: " + key[str(greatestIndex+1)])
    print("Cost: " + str(cost))