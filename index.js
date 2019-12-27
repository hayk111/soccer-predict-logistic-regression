require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LogisticRegression = require('./logistic-regression');
const _ = require('lodash');

const leagueTeams = [ 
  'Bournemouth', 'Cardiff', 'Fulham', 'Crystal Palace', 'Huddersfield', 
  'Chelsea', 'Newcastle', 'Tottenham', 'Watford', 'Brighton', 'Wolves', 
  'Everton', 'Arsenal', 'Man City', 'Liverpool', 'West Ham', 'Southampton', 
  'Burnley', 'Man United', 'Leicester'
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
  iterations: 8,
  batchSize: 3,
  teams: leagueTeams
});

console.log("TCL: features", features.length, testFeatures.length, features[0]);

regression.train();

// console.log('avg:', regression.getHomeTeamMeanResult(features, _.flatMap(labels), 350));

console.log('test:', regression.test(testFeatures, _.flatMap(testLabels)));
