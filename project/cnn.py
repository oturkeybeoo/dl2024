import random

def matrix_convolution(matrix, filter):
  res = 0
  for i in range(len(filter)):
    for j in range(len(filter[i])):
      res += filter[i][j]*matrix[i][j]
  return res

def ceiling(x):
  if int(x) == x:
    return x
  else:
    return int(x+1)

def crop_matrix(img, i, j, size):
  half = int(ceiling(size/2))
  if size % 2 == 0:
    return [row[j+(1-half):j+(1+half)] for row in img[i+(1-half):i+(1+half)]]
  else:
    return [row[j+(1-half):j+half] for row in img[i+(1-half):i+half]]

def deep_copy(img):
  height = len(img)
  width = len(img[0])
  deep_copy = [[0 for _ in range(width)] for _ in range(height)]

  for i in range(height):
    for j in range(width):
        deep_copy[i][j] = img[i][j]

  return deep_copy


def padding(img):
  padded = deep_copy(img)
  for i in range(len(padded)):
    padded[i] = [0] + padded[i] + [0]

  h_padding = [[0 for _ in range(len(padded[i]))]]
  padded = h_padding + padded + h_padding

  return padded

def list_dimensions(l):
    if not isinstance(l, list):
        return 0
    elif not l:
        return 1
    else:
        return 1 + max(list_dimensions(item) for item in l)

def convolution(imgs, filters):
  if list_dimensions(imgs) == 2: imgs = [imgs]
  res = []
  for img in imgs:
    padded = padding(img)
    height = len(padded)
    width = len(padded[0])
    for filter in filters:
      layer = [[0 for _ in range(len(img[0]))] for _ in range(len(img))]

      for i in range(1,height-1):
        for j in range(1,width-1):
          matrix = crop_matrix(padded, i, j, len(filter))
          layer[i-1][j-1] = matrix_convolution(matrix, filter)
      res.append(layer)

  return res

def max_value(l):
  max_value = float('-inf')
  for row in l:
    for value in row:
      if value > max_value: max_value = value
  return max_value

def max_pooling(img, size, stride):
  height = len(img)
  width = len(img[0])

  res_height = int((height-size)/stride + 1)
  res_width = int((width-size)/stride + 1)
  
  res = [[0 for _ in range(res_width)] for _ in range(res_height)]


  res_i = 0
  res_j = 0
  for i in range(res_height):
    for j in range(res_width):
      start_height = i * stride
      start_width = j * stride
      end_height = start_height + size
      end_width = start_width + size

      pool = [row[start_width: end_width] for row in img[start_height: end_height]]
      res[res_i][res_j] = max_value(pool)
      res_j += 1
    res_i += 1
    res_j = 0
  return res

def max_poolings(imgs, size, stride):
  if list_dimensions(imgs) == 2: imgs = [imgs]
  res = []
  for img in imgs:
    res.append(max_pooling(img, size, stride))
  return res

def flatten(list_3d):
  res = []
  for layer in list_3d:
    for row in layer:
      for value in row:
        res.append(value)
  return res

def relu(x):
  if x < 0: return 0
  else: return x

def soft_max(outputs):
  if sum(outputs) == 0: return [0 for _ in range(len(outputs))]
  res = []
  for output in outputs:
    res.append(output/sum(outputs))
  return res

def dense(layer, weights):
  res = []
  for weight in weights:
    output = 0
    for i in range(len(layer)):
      output += layer[i]*weight[i]
    res.append(relu(output))
  res = soft_max(res)
  return res

def output(layer, weights):
  res = []
  for weight in weights:
    output = 0
    for i in range(len(layer)):
      output += layer[i]*weight[i]
    res.append(relu(output))
  res = soft_max(res)
  return res

def predict(result):
  certainty = 0
  prediction = 0
  for i, pred in enumerate(result):
    if pred > certainty:
      certainty = pred
      prediction = i+1
  # print(f'Predict: {prediction}, Certainty: {certainty}')
  return prediction
  
def accuracy(pred, actual):
  corrects = 0
  for p, a in zip(pred, actual):
    if p == a: corrects += 1
  return corrects/len(actual)
 
filter1 = [[-1,0,1],
          [-1,0,1],
          [-1,0,1]]
filter2 = [[-1,-1,-1],
          [0,0,0],
          [1,1,1]]
filter3 = [[-1,-1,0],
          [-1,0,1],
          [0,1,1]]
filter4 = [[0,1,1],
          [1,0,1],
          [1,1,0]]

matrix = [[1,2,3,10],
          [4,5,6,11],
          [7,8,9,12],
          [13,14,15,16]]

x_train = []
with open("D:/Master/Deep Learning/project/x_train1.txt", 'r') as file:
    file_contents = file.read()
    file_contents = file_contents.replace('[', '')
    file_contents = file_contents.replace(']', '')
    xs = file_contents.split('\n\n')

    for x in xs:
      rows = x.split('\n')
      data = []
      for row in rows:
        pixels = row.split(', ')
        pixels = [int(item) for item in pixels]
        data.append(pixels)
      x_train.append(data)

y_train = []
with open("D:/Master/Deep Learning/project/y_train1.txt", 'r') as file:
    file_contents = file.read()
    ys = file_contents.split('\n')
    ys = [int(item) for item in ys]
    y_train = ys

if __name__ == "__main__":
  seed = 3
  random.seed(seed)
  filters = [filter1, filter2, filter3, filter4]
  predictions = []
  for img in x_train:
    layer1 = convolution(img, filters)
    pooling1 = max_poolings(layer1, 5, 3)
    # print("POOLING 1", pooling1[0])
    layer2 = convolution(pooling1, filters)
    pooling2 = max_poolings(layer2, 5, 3)
    # print("POOLING 2", pooling2[0])
    layer3 = convolution(pooling2, filters)
    # print("LAYER 3", layer3[0])
    pooling3 = max_poolings(layer3, 2, 2)
    # print("POOLING 3", pooling3[0])
    flatten1 = flatten(pooling3)
    # print("FLATTEN 1", flatten1)
    weight1 = [[random.uniform(-0.01, 0.01) for _ in range(len(flatten1))] for _ in range(len(flatten1))]
    dense1 = dense(flatten1, weight1)
    # print("DENSE 1", dense1)
    dense2 = dense(dense1, weight1)
    # print("DENSE 2", dense2)
    weight2 = [[random.uniform(-0.01, 0.01) for _ in range(len(dense1))] for _ in range(10)]
    output1 = output(dense2, weight2)
    prediction = predict(output1)
    predictions.append(prediction)

  accuracy_score = accuracy(predictions, y_train)
  print(f"Accuracy: {accuracy_score}")
