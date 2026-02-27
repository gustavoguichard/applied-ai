import tf from '@tensorflow/tfjs-node'

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1] // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ['premium', 'medium', 'basic'] // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1] // basic - Carlos
]

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print()
outputYs.print()

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential()
  // First layer of neural network
  // input shape is 7 positions (idade normalizada, 3 cores, 3 localizações)
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

// const person = { name: 'zé', idade: 28, cor: 'verde', localizacao: 'Curitiba' }
// normalize the person data
// ex: idade_min = 25, idade_max = 40, normalized = (28 - 25) / (40 - 25) = 0.2
const normalizedPersonTenson = [
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

const predictions = await predict(model, normalizedPersonTenson)
console.log(
  predictions
    .sort((a, b) => b.prob - a.prob)
    .map((p) => `${labelsNomes[p.index]} - ${(p.prob * 100).toFixed(2)}%`)
    .join('\n')
)
