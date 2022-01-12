import 'dart:math';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as imageLib;
import 'package:object_detection/tflite/recognition.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

import 'stats.dart';

/// Classifier
class Classifier {
  /// Instance of Interpreter
  Interpreter _interpreter;

  /// Labels file loaded as list
  List<String> _labels;

  static const String MODEL_FILE_NAME = "best2.tflite";
  static const String LABEL_FILE_NAME = "best.txt";

  /// Input size of image (height = width = 300)
  static const int INPUT_SIZE = 320;

  /// Result score threshold
  static const double THRESHOLD = 0.5;

  /// [ImageProcessor] used to pre-process the image
  ImageProcessor imageProcessor;

  /// Padding the image to transform into square
  int padSize;

  /// Shapes of output tensors
  List<List<int>> _outputShapes;

  /// Types of output tensors
  List<TfLiteType> _outputTypes;

  /// Number of results to show
  static const int NUM_RESULTS = 10;

  Classifier({
    Interpreter interpreter,
    List<String> labels,
  }) {
    loadModel(interpreter: interpreter);
    loadLabels(labels: labels);
  }

  /// Loads interpreter from asset
  void loadModel({Interpreter interpreter}) async {
    try {
      _interpreter = interpreter ??
          await Interpreter.fromAsset(
            MODEL_FILE_NAME,
            options: InterpreterOptions()..threads = 4,
          );

      var outputTensors = _interpreter.getOutputTensors();
      _outputShapes = [];
      _outputTypes = [];
      outputTensors.forEach((tensor) {
        _outputShapes.add(tensor.shape);
        _outputTypes.add(tensor.type);
      });
    } catch (e) {
      print("Error while creating interpreter: $e");
    }
  }

  /// Loads labels from assets
  void loadLabels({List<String> labels}) async {
    try {
      _labels =
          labels ?? await FileUtil.loadLabels("assets/" + LABEL_FILE_NAME);
    } catch (e) {
      print("Error while loading labels: $e");
    }
  }


  /// Pre-process the image
  TensorImage getProcessedImage(TensorImage inputImage) {
    padSize = max(inputImage.height, inputImage.width);
    if (imageProcessor == null) {
      imageProcessor = ImageProcessorBuilder()
          .add(ResizeWithCropOrPadOp(padSize, padSize))
          .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeMethod.BILINEAR))
          .add(DequantizeOp(0, 255.0))
          .build();
    }
    inputImage = imageProcessor.process(inputImage);
    return inputImage;
  }


  Map<String, dynamic> predict(imageLib.Image image) {
    var predictStartTime = DateTime.now().millisecondsSinceEpoch;

    if (_interpreter == null) {
      print("Interpreter not initialized");
      return null;
    }

    var preProcessStart = DateTime.now().millisecondsSinceEpoch;

    TensorImage inputImage =  TensorImage(TfLiteType.float32);
    inputImage.loadImage(image);


    inputImage = getProcessedImage(inputImage);

    var preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;

    TensorBuffer outputLocations = TensorBufferFloat([6300, 4]);
    TensorBuffer output = TensorBufferFloat(_outputShapes[0]);
    TensorBuffer outputScores = TensorBufferFloat([6300, 1]);
    TensorBuffer outputObjScores = TensorBufferFloat([6300, 1]);
    List<Object> inputs = [inputImage.buffer];

    // Outputs map
    Map<int, Object> outputs = {
      0: output.buffer,
    };

    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    _interpreter.runForMultipleInputs(inputs, outputs);

    var flatList = output.getDoubleList();
    List<double> outputLocationsList = [];
    List<int> outputScoresList = [];
    List<double> outputObjScoresList = [];

    for (int i = 0; i < 6300; i++) {
      var subList = flatList.sublist(i*6, (i+1)*6);
      outputLocationsList.addAll(subList.sublist(0, 4));
      outputScoresList.addAll([subList.sublist(5).indexOf(subList.sublist(5).reduce(max))]);
      outputObjScoresList.addAll(subList.sublist(4, 5));
    }
    outputLocations.loadList(outputLocationsList, shape: [6300, 4]);
    outputScores.loadList(outputScoresList, shape: [6300, 1]);
    outputObjScores.loadList(outputObjScoresList, shape: [6300, 1]);

    var inferenceTimeElapsed =
        DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;

    // Using bounding box utils for easy conversion of tensorbuffer to List<Rect>
    List<Rect> locations = BoundingBoxUtils.convert(
      tensor: outputLocations,
      valueIndex: [0, 1, 2, 3],
      boundingBoxAxis: 1,
      boundingBoxType: BoundingBoxType.CENTER,
      coordinateType: CoordinateType.RATIO,
      height: INPUT_SIZE,
      width: INPUT_SIZE,
    );

    List<Recognition> recognitions = [];

    for (int i = 0; i < 6300; i++) {
      var score = outputObjScores.getDoubleValue(i);
      var confi = outputScores.getDoubleValue(i);
      // print(confi);
      // print(" ---------------------");
      // print(outputLocations.getDoubleList().sublist(i*4, 4*(i+1)));
      // var labelIndex = outputScores.getDoubleList().sublist(i*80, 80*(i+1)).indexOf(confi);
      var label = _labels.elementAt(confi.toInt());

      if (score > 0.5) {
        print(score);
        print(confi);
        // inverse of rect
        // [locations] corresponds to the image size 300 X 300
        // inverseTransformRect transforms it our [inputImage]
        Rect transformedRect = imageProcessor.inverseTransformRect(
            locations[i], image.height, image.width);

        recognitions.add(
          Recognition(i, label, confi, transformedRect),
        );
      }
    }

    var predictElapsedTime =
        DateTime.now().millisecondsSinceEpoch - predictStartTime;

    return {
      "recognitions": recognitions,
      "stats": Stats(
          totalPredictTime: predictElapsedTime,
          inferenceTime: inferenceTimeElapsed,
          preProcessingTime: preProcessElapsedTime)
    };
  }

  /// Gets the interpreter instance
  Interpreter get interpreter => _interpreter;

  /// Gets the loaded labels
  List<String> get labels => _labels;
}
