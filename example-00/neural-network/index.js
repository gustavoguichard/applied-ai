import tf from '@tensorflow/tfjs-node'

// Example people for training (each person with age, color, and location)
// const people = [
//     { name: "Erick", age: 30, color: "blue", location: "São Paulo" },
//     { name: "Ana", age: 25, color: "red", location: "Rio" },
//     { name: "Carlos", age: 40, color: "green", location: "Curitiba" }
// ];

// Input vectors with already normalized and one-hot encoded values
// Order: [normalized_age, blue, red, green, São Paulo, Rio, Curitiba]
// const peopleTensor = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// We use only numeric data, as the neural network only understands numbers.
// normalizedPeopleTensor corresponds to the model's input dataset.
const normalizedPeopleTensor = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1] // Carlos
]

// Labels for categories to be predicted (one-hot encoded)
// [premium, medium, basic]
const labelNames = ['premium', 'medium', 'basic'] // Label order
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1] // basic - Carlos
]

// We create input (xs) and output (ys) tensors to train the model
const inputXs = tf.tensor2d(normalizedPeopleTensor)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print()
outputYs.print()

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential()
  // First layer of neural network
  // input shape is 7 positions (normalized age, 3 colors, 3 locations)
  // 80 neurons because the training dataset is small
  // The more neurons, the more complex the model is and consequently more proccessing power is needed
  // ReLU is a activation function that is used to introduce non-linearity to the model
  // It returns 0 if the input is negative, and the input if it is positive
  // it filters out the negative values and keeps the positive values
  model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

  // Output layer with 3 neurons because we have 3 labels (premium, medium, basic)
  // softmax is a activation function that is used to output a probability distribution over the 3 labels
  // it returns a probability for each label
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

  // Compile the model
  // Optimizer ADAM which means Adaptive Moment Estimation which is a neural network
  // optimizer that is used to update the weights of the model it learns with the
  // history of right and wrong predictions
  // Loss function is categoricalCrossentropy because we have 3 labels
  // categoricalCrossentropy is a loss function that is used to measure the difference between the predicted and actual labels
  // metrics is an array of metrics to be used to evaluate the model
  // the more far away the predicted label is from the actual label, the higher the loss
  // classic examples: image classification, recommendation systems, label prediction, etc.
  // anything where the answer is a category or label
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  // Train the model
  // verbose disables the internal logging of the model
  // shuffle is a boolean that shuffles the data before each epoch to avoid Bias
  // epochs is the number of times the model will see the data
  await model.fit(inputXs, outputYs, {
    epochs: 100,
    verbose: 0,
    shuffle: true
    // callbacks: {
    //   onEpochEnd: (epoch, logs) => {
    //     console.log(`Epoch ${epoch} - Loss: ${logs.loss}`)
    //   }
    // }
  })
  return model
}

// The more data the better the model will be so the algorithm will be able to predict the labels with more accuracy
const model = await trainModel(inputXs, outputYs)

// const person = { name: 'zé', age: 28, color: 'green', location: 'Curitiba' }
// normalize the person data
// ex: age_min = 25, age_max = 40, normalized = (28 - 25) / (40 - 25) = 0.2
const normalizedPersonTensor = [
  [
    0.2, // Normalized age
    1, // Blue color
    0, // Red color
    0, // Green color
    0, // São Paulo
    1, // Rio
    0 // Curitiba
  ]
]

async function predict(model, person) {
  // transform the person data into a tensor
  const tfInput = tf.tensor2d(person)
  // predict the label
  const prediction = await model.predict(tfInput)
  const predictionArray = await prediction.array()
  return predictionArray[0].map((prob, index) => ({ prob, index }))
}

const predictions = await predict(model, normalizedPersonTensor)
console.log(
  predictions
    .sort((a, b) => b.prob - a.prob)
    .map((p) => `${labelNames[p.index]} - ${(p.prob * 100).toFixed(2)}%`)
    .join('\n')
)
