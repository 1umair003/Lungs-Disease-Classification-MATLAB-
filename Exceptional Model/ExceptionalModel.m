close all
clear all
clc

%% Load the dataset
imdsTrain = imageDatastore('C:/Users/umair/OneDrive/Desktop/MATLAB/Lung Disease Dataset/train', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsValidation = imageDatastore('C:/Users/umair/OneDrive/Desktop/MATLAB/Lung Disease Dataset/val', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore('C:/Users/umair/OneDrive/Desktop/MATLAB/Lung Disease Dataset/test', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

%% Display some training images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 16);
figure
for i = 1:16
    subplot(4, 4, i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end

%% Load pretrained Xception model
net = xception;
%analyzeNetwork(net);

%% Modify network
inputSize = net.Layers(1).InputSize;

%% Extract layers except the last 3
layersTransfer = net.Layers(1:end-3);

%% Determine the number of classes
numClasses = numel(categories(imdsTrain.Labels));

%% Define the new layers for transfer learning
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classification')];

%% Create a layer graph from the original network
lgraph = layerGraph(net);

%% Remove the last 3 layers (fully connected, softmax, and classification layers)
lgraph = removeLayers(lgraph, {'predictions', 'predictions_softmax', 'ClassificationLayer_predictions'});

%% Add the new layers to the layer graph
lgraph = addLayers(lgraph, newLayers);

%% Connect the new layers to the graph
lgraph = connectLayers(lgraph, 'avg_pool', 'new_fc');

%% Visualize the modified network
analyzeNetwork(lgraph);

%% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

%% Use ColorPreprocessing to ensure all images have the same number of channels
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Train the network
netTransfer = trainNetwork(augimdsTrain, lgraph, options);

%% Save the trained model
modelFilename = 'trainedXception.mat';
save(modelFilename, 'netTransfer');

%% Load the trained model
loadedData = load("trainedXception.mat");
netTransferLoaded = loadedData.netTransfer;

%% Classify validation images
[YpredValidation, scoresValidation] = classify(netTransferLoaded, augimdsValidation);

%% Display some validation predictions
idx = randperm(numel(imdsValidation.Files), 4);
figure
for i = 1:4
    subplot(2, 2, i)
    I = readimage(imdsValidation, idx(i));
    imshow(I)
    label = YpredValidation(idx(i));
    title(string(label));
end

%% Calculate validation accuracy
YValidation = imdsValidation.Labels;
validationAccuracy = mean(YpredValidation == YValidation)

%% Plot confusion matrix for validation data
figure
plotconfusion(YValidation, YpredValidation)

%% Classify test images
[YpredTest, scoresTest] = classify(netTransferLoaded, augimdsTest);

%% Calculate test accuracy
YTest = imdsTest.Labels;
testAccuracy = mean(YpredTest == YTest)

%% Plot confusion matrix for test data
figure
plotconfusion(YTest, YpredTest)

%% Display some test predictions
idx = randperm(numel(imdsTest.Files), 4);
figure
for i = 1:4
    subplot(2, 2, i)
    I = readimage(imdsTest, idx(i));
    imshow(I)
    label = YpredTest(idx(i));
    title(string(label));
end
