require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LogisticRegression = require('./logistic-regression');
const _ = require('lodash');

const leagueTeams = [ 'Bournemouth', 'Cardiff', 'Fulham', 'Crystal Palace', 'Huddersfield', 'Chelsea', 'Newcastle', 'Tottenham', 'Watford',
  'Brighton', 'Wolves', 'Everton', 'Arsenal', 'Man City', 'Liverpool', 'West Ham', 'Southampton', 'Burnley', 'Man United', 'Leicester'
];

let { features, labels, testFeatures, testLabels } = loadCSV('./data/18-19.csv', {
  splitTest: 200,
  dataColumns: ['HomeTeam', 'AwayTeam'],
  labelColumns: ['FTR'],
  converters: {
    'FTR': (value) => {
      switch (value) {
        case 'H':
          return [1, 0, 0];
        case 'D':
          return [0, 1, 0];
        case 'A':
          return [0, 0, 1];
      }
    }
  }
});

const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.1,
  iterations: 10,
  batchSize: 3,
  teams: leagueTeams
});

regression.train();

// regression.predict([
//   ['Tottenham', 'Everton'],
//   ['Liverpool', 'Wolves'],
//   ['Huddersfield', 'Man United'],
//   ['Southampton', 'Huddersfield'],
//   ['Wolves', 'Fulham']
// ]).print()

// console.log('test:', regression.test(testFeatures, _.flatMap(testLabels)));

// console.log("TCL: testFeatures", testFeatures, _.flatMap(testLabels))
console.log('test:', regression.test(testFeatures, _.flatMap(testLabels)));
