/**
 * Model Training Web Worker — runs TensorFlow.js model training off the main thread.
 *
 * Web Workers run in a separate thread, so heavy ML computations
 * don't block the UI. Communication with the main thread is via:
 *   - self.onmessage: receives commands from WorkerController
 *   - postMessage(): sends results/progress back to WorkerController
 *
 * Message protocol (matches workerEvents in constants.js):
 *   Inbound:  { action: 'train:model', users: [...], products: [...] }
 *   Inbound:  { action: 'recommend', user: {...} }
 *   Outbound: { type: 'progress:update', progress: { progress: 50 } }
 *   Outbound: { type: 'training:log', epoch, loss, accuracy }
 *   Outbound: { type: 'training:complete' }
 *   Outbound: { type: 'recommend', user, recommendations: [...] }
 *
 * ML Pipeline Overview:
 *
 * 1. FEATURE ENCODING (encodeProduct / encodeUser)
 *    Each product becomes a fixed-length numeric vector:
 *      [normalized_price, avg_buyer_age, ...one_hot_category, ...one_hot_color]
 *    Each feature is scaled by a hand-tuned weight (WEIGHTS) so the model
 *    pays more attention to category (0.4) than color (0.3) than price (0.2) than age (0.1).
 *    A user is encoded as the element-wise mean of their purchased product vectors,
 *    capturing their "average taste profile". Users with no purchases get a fallback
 *    vector based only on their age.
 *
 * 2. TRAINING DATA (createTrainingData)
 *    For every (user, product) pair, we concatenate [userVector, productVector]
 *    as input and label it 1 if the user bought that product, 0 otherwise.
 *    This is a binary classification: "would this user buy this product?"
 *
 * 3. NEURAL NETWORK (configureNeuralNetAndTrain)
 *    A 4-layer sequential model: Dense(128) → Dense(64) → Dense(32) → Dense(1, sigmoid).
 *    Trained with binary cross-entropy loss and the Adam optimizer.
 *    Reports epoch-level metrics (loss, accuracy) back to the UI via postMessage.
 *
 * 4. INFERENCE (recommend)
 *    Encodes the target user, pairs their vector with every product vector,
 *    runs model.predict(), and returns products sorted by predicted purchase
 *    probability (score) in descending order.
 *
 * _globalCtx holds shared state (normalization bounds, lookup indices,
 * pre-encoded product vectors) between the train and recommend phases.
 */

// TensorFlow.js is loaded from CDN — exposes the global `tf` namespace.
// Must be imported before any tf.* calls.
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js'
import { workerEvents } from '../events/constants.js'

console.log('Model training worker initialized')

// --------------------------------------------------------------------------
// GLOBAL STATE
// --------------------------------------------------------------------------

// _globalCtx: After training, holds all the context needed for inference:
//   - products: the full product catalog
//   - productVectors: pre-encoded product tensors (avoids re-encoding at inference)
//   - categoriesIndex / colorsIndex: maps category/color strings → integer indices for one-hot
//   - minAge, maxAge, minPrice, maxPrice: normalization bounds
//   - numCategories, numColors: sizes for one-hot encoding
//   - dimentions: total feature vector length per product/user
//   - normalizedProductAvgAge: per-product average buyer age (normalized 0–1)
let _globalCtx = {}

// _model: The trained tf.Sequential model. null until training completes.
// Used by recommend() to run inference.
let _model = null

// Feature importance weights — hand-tuned multipliers applied to each feature
// dimension before concatenation. Higher weight = the model "sees" bigger
// differences in that feature, making it more influential in predictions.
// Category dominates (0.4) because "what kind of product" is the strongest
// purchase signal; color (0.3) captures aesthetic preference; price (0.2) and
// buyer age profile (0.1) are weaker but still informative signals.
const WEIGHTS = {
  category: 0.4,
  color: 0.3,
  price: 0.2,
  age: 0.1
}

// --------------------------------------------------------------------------
// UTILITY FUNCTIONS
// --------------------------------------------------------------------------

// Min-max normalization: maps a value from [min, max] → [0, 1].
// The `|| 1` guard prevents division by zero when min === max
// (all values identical), collapsing everything to 0 in that edge case.
function normalize(value, min, max) {
  return (value - min) / (max - min || 1)
}

// Returns { min, max } for a numeric array — used for normalization bounds on ages and prices
function bounds(values) {
  return { min: Math.min(...values), max: Math.max(...values) }
}

// Maps an array of unique items to { item: index } — used for one-hot encoding of categories and colors
function buildIndex(items) {
  return Object.fromEntries(items.map((item, i) => [item, i]))
}

// Arithmetic mean of a numeric array — used for average age computation and buyer age fallback
function mean(values) {
  return values.reduce((sum, v) => sum + v, 0) / values.length
}

function groupBuyerAgesByProduct(users, products) {
  const buyerAgesPerProduct = new Map()
  users.forEach((user) => {
    user.purchases.forEach((p) => {
      const purchasedProduct = products.find((product) => product.id === p)
      if (purchasedProduct) {
        const agesForProduct =
          buyerAgesPerProduct.get(purchasedProduct.name) ?? []
        agesForProduct.push(user.age)
        buyerAgesPerProduct.set(purchasedProduct.name, agesForProduct)
      }
    })
  })
  return buyerAgesPerProduct
}

// --------------------------------------------------------------------------
// CONTEXT BUILDING — makeContext()
// --------------------------------------------------------------------------

// Builds the shared context object from raw users and products data.
// This is the first step of the pipeline: it precomputes all the lookup
// tables, normalization bounds, and statistics that every other function needs.
//
// Returns a context object with:
//   - categoriesIndex: { "Electronics": 0, "Clothing": 1, ... }
//   - colorsIndex:     { "Black": 0, "Red": 1, ... }
//   - min/max for age and price (for normalization)
//   - numCategories / numColors (for one-hot vector lengths)
//   - dimentions: total feature vector size = 2 (price + age) + numCategories + numColors
//   - normalizedProductAvgAge: for each product, the normalized average age
//     of users who bought it (used as a feature to capture age-demographic affinity)
function makeContext({ users, products }) {
  // Extract all user ages and product prices to compute normalization bounds
  const ages = users.map((user) => user.age)
  const ageBounds = bounds(ages)
  const priceBounds = bounds(products.map((product) => product.price))

  // Build category/color → integer index mappings for one-hot encoding.
  // Example: if products have categories ["Electronics", "Clothing", "Books"],
  // categoriesIndex produces { "Electronics": 0, "Clothing": 1, "Books": 2 }
  const uniqueCategories = [
    ...new Set(products.map((product) => product.category))
  ]
  const uniqueColors = [...new Set(products.map((product) => product.color))]
  const categoriesIndex = buildIndex(uniqueCategories)
  const colorsIndex = buildIndex(uniqueColors)

  // Mean age across all users — used as fallback for products nobody has bought yet
  const meanAge = mean(ages)

  const buyerAgesPerProduct = groupBuyerAgesByProduct(users, products)

  // For each product, compute the average buyer age and normalize it to [0, 1].
  // Products with no purchases fall back to the global meanAge.
  // This normalized value becomes one of the product's features (the "age" dimension).
  const normalizedProductAvgAge = Object.fromEntries(
    products.map((product) => {
      const ages = buyerAgesPerProduct.get(product.name)
      const avgAge = ages ? mean(ages) : meanAge
      return [product.name, normalize(avgAge, ageBounds.min, ageBounds.max)]
    })
  )

  return {
    users,
    products,
    colorsIndex,
    categoriesIndex,
    minAge: ageBounds.min,
    maxAge: ageBounds.max,
    minPrice: priceBounds.min,
    maxPrice: priceBounds.max,
    numCategories: uniqueCategories.length,
    numColors: uniqueColors.length,
    // Total feature vector length: 1 (price) + 1 (age) + numCategories + numColors
    // Example: 2 scalar features + 5 categories + 4 colors = 11 dimensions
    dimentions: 2 + uniqueCategories.length + uniqueColors.length,
    normalizedProductAvgAge
  }
}

// --------------------------------------------------------------------------
// FEATURE ENCODING — encodeProduct() / encodeUser()
// --------------------------------------------------------------------------

// Creates a weighted one-hot vector.
// Standard one-hot: [0, 0, 1, 0] — the "1" is at position `index`.
// Weighted one-hot: [0, 0, 0.4, 0] — the "1" is replaced by the weight.
// This lets us control how much influence this feature has relative to others
// without needing a separate normalization layer in the model.
// tf.oneHot returns a tensor of dtype int32. Casting to float32 is necessary so we can correctly apply the weight multiplier (which may be a non-integer) in later operations.
const oneHotWeighted = (index, length, weight) => {
  return tf.oneHot(index, length).cast('float32').mul(weight)
}

// Encodes a single product into a fixed-length numeric feature vector.
//
// The resulting vector layout (concatenated in order):
//   [price_normalized * 0.2, avg_buyer_age_normalized * 0.1, ...category_one_hot * 0.4, ...color_one_hot * 0.3]
//    \___ 1 float ___/        \___ 1 float ___/               \__ numCategories floats __/ \__ numColors floats __/
//
// Total length = context.dimentions = 2 + numCategories + numColors
//
// Each feature is pre-multiplied by its WEIGHT so the model doesn't need to
// learn the relative importance from scratch — we inject domain knowledge.
function encodeProduct(product, context) {
  // Normalized price in [0, 1], scaled by price weight (0.2)
  // Why do we use tensor1d?
  // tf.tensor1d creates a 1-dimensional tensor (vector) from a flat array, which is the required input type for later tensor operations like tf.concat1d.
  // Using tensor1d ensures this feature segment can be concatenated with others (e.g., category and color one-hot vectors) in encodeProduct().
  const price = tf.tensor1d([
    normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price
  ])
  // Average buyer age for this product, normalized in [0, 1], scaled by age weight (0.1).
  // Falls back to 0.5 (midpoint) if the product has no purchase history via ?? operator.
  const age = tf.tensor1d([
    (context.normalizedProductAvgAge[product.name] ?? 0.5) * WEIGHTS.age
  ])
  // One-hot category vector with weight 0.4 applied to the active position
  const category = oneHotWeighted(
    context.categoriesIndex[product.category],
    context.numCategories,
    WEIGHTS.category
  )
  // One-hot color vector with weight 0.3 applied to the active position
  const color = oneHotWeighted(
    context.colorsIndex[product.color],
    context.numColors,
    WEIGHTS.color
  )
  // Concatenate all feature segments into a single 1D vector
  // this is equivalent to [price, age, ...category, ...color] but as a tensor.
  // It creates a 1D tensor with the same layout and order as that array.
  return tf.concat1d([price, age, category, color])
}

function encodeUserFromPurchases(user, context) {
  // Imagine a 1D tensor (vector) like this:
  //   [a, b, c]
  // If we "stack" three such vectors, we get a 2D tensor (matrix):
  //   [[a, b, c],
  //    [a, b, c],
  //    [a, b, c]]
  // In tf.stack(arr), each element of arr is a 1D tensor,
  // and the result is a 2D tensor where each row is one of those 1D tensors.
  return (
    tf
      .stack(
        // For each purchased product ID in user.purchases...
        user.purchases.map((purchase) => {
          // Find the full product object in the products array using the ID
          const product = context.products.find((p) => p.id === purchase)
          // If the product exists, encode it to a feature vector tensor
          if (product) {
            return encodeProduct(product, context)
          }
          // If the product ID isn't valid, return a zero vector of the correct dimension
          // tf.zeros([n]) creates a 1D tensor of shape [n] filled with zeros.
          // Here, we use it to produce a "blank" feature vector (all 0s)
          // if the product ID is invalid/missing, so it doesn't pollute the user's profile.
          return tf.zeros([context.dimentions])
        })
      )
      // Compute the mean vector along axis 0 (average each feature column across purchases)
      .mean(0)
      // Reshape the mean 1D tensor into a 2D tensor with shape [1, dimentions]
      .reshape([1, context.dimentions])
  )
}

/**
 * Encodes a "cold start" user into a feature vector tensor for the model.
 * Cold start users have no purchase history, so we cannot compute their
 * average taste profile from purchases. Instead, we create a vector using only their age.
 *
 * The resulting vector is shaped as:
 *   [0 (price placeholder), normalized_age * age_weight, ...zero_category, ...zero_color]
 * This gives age *some* signal, but categories and colors are all zeroed out.
 *
 * Returns:
 *   Tensor of shape [1, dimentions]
 */
function encodeColdStartUser(user, context) {
  // Create a tensor made up of:
  // - tf.zeros([1]): placeholder for price feature (no price info for users)
  // - normalized age ([0,1]), scaled by age weight
  // - zeros for categories (one-hot, all off)
  // - zeros for colors (one-hot, all off)
  return (
    tf
      .concat1d([
        tf.zeros([1]), // price: no value for user profile
        tf
          .tensor1d([normalize(user.age, context.minAge, context.maxAge)]) // age, normalized [0-1]
          .mul(WEIGHTS.age), // apply feature weight as with product encoding
        tf.zeros([context.numCategories]), // all categories "off"
        tf.zeros([context.numColors]) // all colors "off"
      ])
      // At this point, we have a 1D tensor of shape [dimentions].
      // We reshape it to a 2D tensor with shape [1, dimentions], i.e., an array containing one array of features:
      // from [dimentions] to [[dimentions]].
      .reshape([1, context.dimentions])
  )
}

/**
 * Encodes a user (with or without purchase history) into a [1, dimentions] tensor.
 * If the user has made purchases, computes their taste profile based on the purchased products.
 * If not, falls back to the cold start encoding (just age).
 */
function encodeUser(user, context) {
  // Use the full encoder if the user has made purchases;
  // otherwise, fall back to cold start logic
  return user.purchases.length > 0
    ? encodeUserFromPurchases(user, context)
    : encodeColdStartUser(user, context)
}

// --------------------------------------------------------------------------
// TRAINING DATA GENERATION — createTrainingData()
// --------------------------------------------------------------------------

function createUserProductPairs(user, context) {
  // Encode the user into a flat vector (typed array) of shape [dimentions]
  const userVector = encodeUser(user, context).dataSync()

  // For each product
  return context.products.map((product) => {
    // Encode the product into a flat vector (typed array) of shape [dimentions]
    const productVector = encodeProduct(product, context).dataSync()
    // Determine label: 1 if user purchased this product, else 0 (binary classification)
    const label = user.purchases.includes(product.id) ? 1 : 0
    // Create the input feature (concatenate user and product vectors), pair with label
    return { input: [...userVector, ...productVector], label }
  })
}

/**
 * Generates the training dataset for the neural network.
 *
 * This function constructs labeled examples for supervised learning. Each example pairs a user with a product,
 * encoding both into numerical feature vectors and concatenating them.
 *
 * For every user who has made at least one purchase, we generate input-label pairs for all products:
 *   - The input is a concatenation of the user's feature vector and the product's feature vector.
 *   - The label is 1 if the user actually purchased the product ("positive" example), or 0 if not ("negative" example).
 *
 * The labels are essential: during training, the model learns to predict this label — essentially, to estimate the probability
 * that a given user would buy a given product. Without labels, the model could not learn the concept of "purchased" vs. "not purchased",
 * and no recommendation signal would emerge.
 *
 * The resulting structure:
 *   - xs: A 2D tensor of shape [number_of_pairs, 2 * feature_dimensions], holding the inputs.
 *     (Here, number_of_pairs equals the total number of (user, product) combinations for all users with at least one purchase.
 *      In other words, for each such user, we generate a pair for every product in the catalog:
 *      number_of_pairs = sum over all users with purchases of (number of products).)
 *   - ys: A 2D tensor of shape [number_of_pairs, 1], holding the binary labels for each input pair.
 *   - inputDimensions: The size of each input vector (user + product features).
 */
function createTrainingData(context) {
  // Consider only users with at least one purchase (we can't learn from users with no history)
  const pairs = context.users
    .filter((user) => user.purchases.length > 0) // Filter out cold-start users
    .flatMap((user) => createUserProductPairs(user, context)) // Create (user, product) pairs

  // Gather input features and ground truth labels
  const inputs = pairs.map((p) => p.input) // shape: [#pairs, 2 * feature_dimensions]
  const outputs = pairs.map((p) => p.label) // shape: [#pairs]

  // Return tensors and metadata for model training
  return {
    xs: tf.tensor2d(inputs),
    // The second parameter ([outputs.length, 1]) explicitly sets the tensor shape.
    // This ensures that 'ys' is a column vector (shape [N, 1]), rather than a flat [N] vector,
    // which is required when fitting a model with a single output neuron.
    ys: tf.tensor2d(outputs, [outputs.length, 1]),
    // inputDimensions is the length of each input vector to the model,
    // which is the concatenation of a user feature vector and a product feature vector.
    // Each is of length `context.dimentions`, so their concatenation is `context.dimentions * 2`.
    inputDimensions: context.dimentions * 2
  }
}

// --------------------------------------------------------------------------
// NEURAL NETWORK — configureNeuralNetAndTrain()
// --------------------------------------------------------------------------

/**
 * Builds and compiles a neural network model for user-product recommendation.
 *
 * This model is designed for binary classification: given feature vectors describing a user and a product,
 * it predicts the probability that the user would purchase the product.
 *
 * Architecture (using TensorFlow.js):
 *   - tf.sequential(): A simple neural network where each layer feeds only into the next.
 *   - Layer 1: Dense (fully connected), 128 neurons, ReLU activation. Receives the input vector shape.
 *   - Layer 2: Dense, 64 neurons, ReLU activation.
 *   - Layer 3: Dense, 32 neurons, ReLU activation.
 *   - Output Layer: Dense, 1 neuron, Sigmoid activation (outputs value between 0 and 1 = probability of purchase).
 *
 * Notes on API:
 *   - tf.layers.dense({
 *       units: NUMBER,            // how many "neurons" in this layer (size of output vector)
 *       inputShape: [N],          // only for first layer: shape of input vectors to the model (how many features)
 *       activation: 'relu'        // activation function; 'relu' introduces nonlinearity
 *     })
 *   - 'sigmoid' activation in the final layer is required for binary outcome (purchase or not).
 *   - .compile() prepares the model for training:
 *       - optimizer: how gradient descent is performed (adam is a common, adaptive choice)
 *       - loss: what error function to minimize (binaryCrossentropy for binary classification)
 *       - metrics: log 'accuracy' during training (fraction of correct predictions)
 */
function buildModel(inputDimensions) {
  // Create a simple sequential neural network model (a stack of layers, each feeding into the next)
  const model = tf.sequential()

  // First hidden layer: takes input of shape [inputDimensions], outputs 128 features, uses ReLU activation
  model.add(
    tf.layers.dense({
      units: 128, // number of "neurons" (outputs) in this layer
      inputShape: [inputDimensions], // the length of each input vector
      activation: 'relu' // ReLU = max(0, x), helps model learn nonlinear relationships
    })
  )

  // Second hidden layer: 64 "neurons", ReLU activation
  model.add(
    tf.layers.dense({
      units: 64,
      activation: 'relu'
    })
  )

  // Third hidden layer: 32 "neurons", ReLU activation
  model.add(
    tf.layers.dense({
      units: 32,
      activation: 'relu'
    })
  )

  // Output layer: 1 neuron (for binary prediction), Sigmoid squashes output to range [0, 1]
  model.add(
    tf.layers.dense({
      units: 1,
      activation: 'sigmoid' // outputs probability of purchase between 0 and 1
    })
  )

  // Compile the model:
  // - optimizer controls how weights are updated (adam is a good adaptive default)
  // - loss is binary crossentropy, suitable for binary classification
  // - metrics: log accuracy during training
  model.compile({
    optimizer: tf.train.adam(0.01), // 0.01 = learning rate; smaller = slower but safer learning
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  })

  return model
}

// Trains a neural network model using the provided training data.
// - trainingData: { xs, ys, inputDimensions }
//   - xs: input feature tensors (user-product pairs as vectors)
//   - ys: labels (1 if user purchased product, 0 otherwise)
//   - inputDimensions: the width of the input layer (xs shape[1])
// This function builds the model, fits it on the training data, and
// posts epoch-level logs (loss & accuracy) back to the main thread.
// Returns: the trained model instance.
async function configureNeuralNetAndTrain(trainingData) {
  // Build a new model using the specified input dimensions
  const model = buildModel(trainingData.inputDimensions)

  // Train the model using the provided xs (inputs) and ys (labels)
  // - epochs: number of times to iterate over the entire dataset
  // - batchSize: number of samples per parameter update
  // - shuffle: randomize data order each epoch (helps generalization)
  // - callbacks: hook for posting progress to main/UI thread after each epoch
  await model.fit(trainingData.xs, trainingData.ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      // onEpochEnd runs after each training epoch completes
      onEpochEnd: (epoch, logs) => {
        // logs.loss = binary cross-entropy loss (how wrong the model is)
        // logs.acc  = accuracy metric (fraction correct)
        // Post progress to the main thread (so UI can update graphs etc.)
        postMessage({
          type: workerEvents.trainingLog,
          epoch: epoch,
          loss: logs.loss,
          accuracy: logs.acc // logs.acc is the default accuracy metric name for binary
        })
      }
    }
  })

  // Return the trained model for use in inference (recommendations)
  return model
}

// --------------------------------------------------------------------------
// PIPELINE ORCHESTRATOR — trainModel()
// --------------------------------------------------------------------------

function preEncodeProducts(products, context) {
  return products.map((product) => ({
    name: product.name,
    meta: { ...product },
    vector: encodeProduct(product, context).dataSync()
  }))
}

async function trainModel({ users, products }) {
  console.log('Training model with users:', users)

  // Notify the main thread that training initialization is halfway done
  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } })

  // Build ML context — includes feature encodings, lookup indices, normalization bounds, etc.
  const context = makeContext({ users, products })

  // Pre-encode all products up front for efficiency during both training & inference
  context.productVectors = preEncodeProducts(products, context)

  // Save the context globally for later inference/recommendation
  _globalCtx = context

  // Assemble the full training dataset (xs: features, ys: labels) according to the constructed context
  const trainingData = createTrainingData(context)

  // Train a fresh neural network using the training data and await until complete
  _model = await configureNeuralNetAndTrain(trainingData)

  // Notify UI/main thread that training reached 100% completion
  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 100 }
  })
  // Signal end of training; main thread may now use the model for inference
  postMessage({ type: workerEvents.trainingComplete })
}

// --------------------------------------------------------------------------
// INFERENCE — recommend()
// --------------------------------------------------------------------------

// Generates personalized product recommendations for a single user.
//
// How it works:
//   1. Encode the user into a feature vector (same space as products)
//   2. For each product, concatenate [userVector, productVector] (same format as training)
//   3. Run all pairs through model.predict() in a single batch (efficient)
//   4. model.predict() returns a Float32Array of sigmoid probabilities via .dataSync()
//   5. Pair each score with the product's metadata (name, price, etc.)
//   6. Sort by score descending — highest purchase probability first
//   7. Post the ranked recommendations back to the main thread
//
// The ctx parameter is _globalCtx, which contains the pre-encoded productVectors
// and all normalization bounds from the training phase.
function recommend(user, ctx) {
  // Guard: if the model hasn't been trained yet, silently skip
  if (!_model) return

  // Encode the user into a feature vector and extract raw floats
  const userVector = encodeUser(user, ctx).dataSync()

  // Build the input batch: one row per product, each row = [userVector, productVector]
  // This mirrors exactly how training data was structured in createTrainingData()
  const inputs = ctx.productVectors.map((product) => [
    ...userVector,
    ...product.vector
  ])

  // Create a 2D tensor and run inference in a single batch (much faster than one-by-one)
  const inputTensor = tf.tensor2d(inputs)
  // .predict() returns a tensor; .dataSync() extracts raw Float32Array values
  const predictions = _model.predict(inputTensor).dataSync()

  // Map prediction scores back to product metadata and sort by score descending.
  // predictions[i] corresponds to ctx.productVectors[i] (same order).
  const recommendations = ctx.productVectors
    .map((product, index) => ({
      ...product.meta, // full product object (id, name, price, category, color, etc.)
      score: predictions[index] // sigmoid probability in [0, 1]
    }))
    .sort((a, b) => b.score - a.score)

  // Send ranked recommendations back to the main thread.
  // WorkerController will emit a DOM event that ProductController listens to.
  postMessage({ type: workerEvents.recommend, user, recommendations })
}

// --------------------------------------------------------------------------
// MESSAGE ROUTING
// --------------------------------------------------------------------------

// Action router — maps incoming action names to handler functions.
// When the worker receives a message with { action: 'train:model', ...data },
// it looks up 'train:model' in this map and calls trainModel(data).
const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx)
}

// Entry point for all messages from WorkerController.
// Every message must have an `action` field matching a key in `handlers`.
// The rest of the message payload is spread into `data` and passed to the handler.
self.onmessage = (e) => {
  const { action, ...data } = e.data
  if (handlers[action]) handlers[action](data)
}
