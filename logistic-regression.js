const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, batchSize: 10, decesionBoundary: 0.5, teams: [] }, 
      options
    );
    
    this.teamPoints = this.initTeamPoints();

    this.features = this.processFeatures(features, labels);
    this.labels = tf.tensor(labels);
    this.costHistory = [];

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]); // initial [[0], [0]] for [[b], [m]]
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    ); 

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        const featureSlice = this.features.slice(
          [startIndex, 0], 
          [batchSize, -1]
        );

        const labelSlice = this.labels.slice(
          [startIndex, 0], 
          [batchSize, -1]
        );
        this.gradientDescent(featureSlice, labelSlice);
      }
      this.recordCost();
      this.updateLearningRate();
    }
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    const incorrect = predictions
      .notEqual(testLabels)
      .sum()
      .get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1); // largest values across horizontal axis
  }

  processFeatures(features, labels) {
    console.log('Man United after 38 game: ', this.getTeamPoints('Man United', features, labels, 38));

    
    // features = tf.tensor(features);
    
    // if (this.mean && this.variance) {
    //   features = features.sub(this.mean).div(this.variance.pow(0.5));
    // } else {
    //   features = this.standardize(features);
    // }
    
    // features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return tf.tensor([[0], [0], [8]]);
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);
    variance.print()``
    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  recordCost() {
    const guesses = this.features.matMul(this.weights).softmax();

    const termOne = this.labels.transpose().matMul(guesses.log());

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses
          .mul(-1)
          .add(1)
          .log()
      );

    const cost = termOne.add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .get(0, 0);

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 0.5;
    }
  }

  getTeamPoints(teamName, features, labels, game = 0) {
    let teamPoints = new Map();
    teamPoints = this.initTeamPoints();

    const featuresPlayed = _.slice(features, 0, game);
    const labelsPlayed = _.slice(labels, 0, game);

    for(let i = 0; i < featuresPlayed.length; i++) {
      switch(labelsPlayed[i].indexOf(1)) {
        case 0:
          console.log("TCL: LogisticRegression -> getTeamPoints -> featuresPlayed[i][1]", featuresPlayed[i][1])
          teamPoints.set(featuresPlayed[i][0], teamPoints.get(featuresPlayed[i][0]) + 3);
          break;
        case 1:
          console.log("TCL: LogisticRegression -> getTeamPoints -> featuresPlayed[i][0]", featuresPlayed[i][0])
          console.log("TCL: LogisticRegression -> getTeamPoints -> featuresPlayed[i][1]", featuresPlayed[i][1])
          teamPoints.set(featuresPlayed[i][0], teamPoints.get(featuresPlayed[i][0]) + 1);
          teamPoints.set(featuresPlayed[i][1], teamPoints.get(featuresPlayed[i][1]) + 1);
          break;
        case 2:
          console.log("TCL: LogisticRegression -> getTeamPoints -> featuresPlayed[i][0]", featuresPlayed[i][0])
          teamPoints.set(featuresPlayed[i][1], teamPoints.get(featuresPlayed[i][1]) + 3);
      }
    }
  
    return teamPoints.get(teamName);
  }

  initTeamPoints() {
    const teamPoints = new Map();
    _.each(this.options.teams, team => teamPoints.set(team, 0));

    return teamPoints;
  }
}

module.exports = LogisticRegression;