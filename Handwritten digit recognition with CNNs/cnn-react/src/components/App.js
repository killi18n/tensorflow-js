import React, { Component } from "react";
import { tidy, browser, sequential, layers, train } from "@tensorflow/tfjs";
import { visor, show, metrics, render } from "@tensorflow/tfjs-vis";
import { MnistData } from "../data/data";

const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine"
];

class App extends Component {
  componentDidMount() {
    this.run();
  }

  showExamples = async data => {
    const surface = visor().surface({
      name: "Input Data Examples",
      tab: "Input Data"
    });

    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    for (let i = 0; i < numExamples; i++) {
      const imageTensor = tidy(() => {
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([28, 28, 1]);
      });

      const canvas = document.createElement("canvas");
      canvas.width = 28;
      canvas.height = 28;
      canvas.style = "margin: 4px;";

      await browser.toPixels(imageTensor, canvas);
      surface.drawArea.appendChild(canvas);
      imageTensor.dispose();
    }
  };

  getModel = () => {
    const model = sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    model.add(
      layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );

    model.add(layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(
      layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: "relu",
        kernelInitializer: "varianceScaling"
      })
    );
    model.add(layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(layers.flatten());

    const NUM_OUTPUT_CLASSES = 10;
    model.add(
      layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: "varianceScaling",
        activation: "softmax"
      })
    );

    const optimizer = train.adam();
    model.compile({
      optimizer,
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });

    return model;
  };

  train = async (model, data) => {
    try {
      const metrics = ["loss", "val_loss", "acc", "val_acc"];
      const container = {
        name: "Model Training",
        styles: { height: "1000px" }
      };
      const fitCallbacks = show.fitCallbacks(container, metrics);

      const BATCH_SIZE = 512;
      const TRAIN_DATA_SIZE = 5500;
      const TEST_DATA_SIZE = 1000;

      const [trainXs, trainYs] = tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
      });

      const [testXs, testYs] = tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
      });

      return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
      });
    } catch (e) {
      console.log(e);
    }
  };

  doPrediction = (model, data, testDataSize = 500) => {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;

    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([
      testDataSize,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      1
    ]);
    const labels = testData.labels.argMax([-1]);
    const preds = model.predict(testxs).argMax([-1]);

    testxs.dispose();
    return [preds, labels];
  };

  showAccuracy = async (model, data) => {
    const [preds, labels] = this.doPrediction(model, data);
    const classAccuracy = await metrics.perClassAccuracy(labels, preds);
    const container = { name: "Accuracy", tab: "Evaluation" };
    show.perClassAccuracy(container, classAccuracy, classNames);
    labels.dispose();
  };

  showConfusion = async (model, data) => {
    const [preds, labels] = this.doPrediction(model, data);
    const confusionMatrix = await metrics.confusionMatrix(labels, preds);
    const container = { name: "Confusion Matrix", tab: "Evaluation" };
    render.confusionMatrix(
      container,
      {
        values: confusionMatrix
      },
      classNames
    );
    labels.dispose();
  };

  run = async () => {
    try {
      const data = new MnistData();
      await data.load();
      await this.showExamples(data);

      const model = this.getModel();
      show.modelSummary({ name: "Model Architecture" }, model);
      await this.train(model, data);
      await this.showAccuracy(model, data);
      await this.showConfusion(model, data);
    } catch (e) {
      console.log(e);
    }
  };

  render() {
    return <div />;
  }
}

export default App;
