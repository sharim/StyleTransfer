import os


image = 'images/content/autumn.jpg'
savePath = 'images/generate/colorful/'
os.chdir('D:/MyProject/StyleTransfer')
i = 0
for dir in os.listdir('models/checkpoints/colorful'):
    i += 1000
    j = i if i <= 39000 else i - 39000
    k = 0 if i <= 39000 else 1
    modelPath = 'models/checkpoints/colorful' + '/' + dir
    os.system('python src/main.py eval --content-image {} --model {} --generate-image {} --cuda 1'.format(image, modelPath, savePath + 'autumn_cf_' + str(k) + '_' + str(j) + '.jpg'))
