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
  // Extract all user ages to compute normalization bounds
  const ages = users.map((user) => user.age)
  const minAge = Math.min(...ages)
  const maxAge = Math.max(...ages)

  // Extract all product prices to compute normalization bounds
  const prices = products.map((product) => product.price)
  const minPrice = Math.min(...prices)
  const maxPrice = Math.max(...prices)

  // Build category → integer index mapping for one-hot encoding.
  // Example: if products have categories ["Electronics", "Clothing", "Books"],
  // this produces { "Electronics": 0, "Clothing": 1, "Books": 2 }
  const categories = new Set(products.map((product) => product.category))
  const categoriesIndex = Object.fromEntries(
    Array.from(categories).map((category, index) => [category, index])
  )

  // Same for colors — each unique color gets an integer index
  const colors = new Set(products.map((product) => product.color))
  const colorsIndex = Object.fromEntries(
    Array.from(colors).map((color, index) => [color, index])
  )

  // Mean age across all users — used as fallback for products nobody has bought yet
  const meanAge = ages.reduce((sum, age) => sum + age, 0) / ages.length

  // Mean price (computed but not currently used; available for future features)
  const meanPrice =
    prices.reduce((sum, price) => sum + price, 0) / prices.length

  // Compute per-product average buyer age.
  // For each product, we sum the ages of all users who bought it and count them.
  // This captures the demographic profile of each product: e.g., "young people
  // buy this product" vs "older people buy this product".
  const ageSums = {}
  const ageCounts = {}

  // Iterate all users and their purchases (which are product IDs).
  // For each purchased product, accumulate the buyer's age into ageSums
  // and increment ageCounts.
  users.forEach((user) => {
    user.purchases.forEach((p) => {
      // Resolve the product ID to the full product object
      const purchasedProduct = products.find((product) => product.id === p)
      if (purchasedProduct) {
        // Running sum of buyer ages for this product
        ageSums[purchasedProduct.name] =
          (ageSums[purchasedProduct.name] || 0) + user.age
        // Running count of buyers for this product
        ageCounts[purchasedProduct.name] =
          (ageCounts[purchasedProduct.name] || 0) + 1
      }
    })
  })

  // For each product, compute the average buyer age and normalize it to [0, 1].
  // Products with no purchases fall back to the global meanAge.
  // This normalized value becomes one of the product's features (the "age" dimension).
  const normalizedProductAvgAge = Object.fromEntries(
    products.map((product) => {
      const avgAge =
        ageCounts[product.name] > 0
          ? ageSums[product.name] / ageCounts[product.name]
          : meanAge
      return [product.name, normalize(avgAge, minAge, maxAge)]
    })
  )

  return {
    users,
    products,
    colorsIndex,
    categoriesIndex,
    minAge,
    maxAge,
    minPrice,
    maxPrice,
    numCategories: categories.size,
    numColors: colors.size,
    // Total feature vector length: 1 (price) + 1 (age) + numCategories + numColors
    // Example: 2 scalar features + 5 categories + 4 colors = 11 dimensions
    dimentions: 2 + categories.size + colors.size,
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
// tf.oneHot returns int32, so we cast to float32 before multiplying by weight.
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
  return tf.concat1d([price, age, category, color])
}

// Encodes a user into a feature vector of the same shape as a product vector.
// This is critical: user and product vectors must live in the same feature space
// so we can concatenate them as input to the neural network.
//
// Strategy:
//   - Users WITH purchases: encode each purchased product, then take the
//     element-wise mean. This creates a "centroid" — an average taste profile.
//     A user who bought an Electronics item and a Clothing item will have
//     ~0.2 in both category slots, capturing their mixed preferences.
//   - Users WITHOUT purchases (cold start): we can't derive taste from history,
//     so we create a sparse vector with only the age feature filled in.
//     Price = 0 (no signal), category/color = all zeros (no preference).
//     The model can still make age-based predictions (e.g., "young users tend to buy X").
function encodeUser(user, context) {
  if (user.purchases.length) {
    // For each purchase ID, find the product and encode it.
    // tf.stack() turns N vectors into a 2D tensor (N rows × dimentions cols).
    // .mean(0) collapses along axis 0, producing the element-wise average
    // across all purchased product vectors — the user's "taste centroid".
    // .reshape([1, dimentions]) adds a batch dimension for model compatibility.
    return tf
      .stack(
        user.purchases.map((purchase) => {
          const product = context.products.find((p) => p.id === purchase)
          if (product) {
            return encodeProduct(product, context)
          }
          // If a purchase ID doesn't match any product (stale data), use a zero vector
          return tf.zeros([context.dimentions])
        })
      )
      .mean(0)
      .reshape([1, context.dimentions])
  } else {
    // Cold-start fallback: no purchases to derive preferences from.
    // Build a vector with: [0 (no price info), normalized_age * weight, 0...0 (no category/color)]
    // This gives the model at least the user's age demographic to work with.
    return tf
      .concat1d([
        tf.zeros([1]), // price slot: zero (no purchase price signal)
        tf
          .tensor1d([normalize(user.age, context.minAge, context.maxAge)])
          .mul(WEIGHTS.age), // age slot: normalized and weighted
        tf.zeros([context.numCategories]), // category slots: all zero (no preference)
        tf.zeros([context.numColors]) // color slots: all zero (no preference)
      ])
      .reshape([1, context.dimentions])
  }
}

// --------------------------------------------------------------------------
// TRAINING DATA GENERATION — createTrainingData()
// --------------------------------------------------------------------------

// Builds the training dataset for supervised learning.
//
// For each user who has at least one purchase, we create a training example
// for EVERY product in the catalog:
//   - Input: [userVector, productVector] concatenated (length = dimentions * 2)
//   - Label: 1 if the user bought this product, 0 if not
//
// This means for N users (with purchases) and M products, we get N × M examples.
// Most labels will be 0 (users buy few of all available products), making this
// an imbalanced binary classification problem — the model learns to distinguish
// "likely to buy" from the sea of "unlikely to buy".
//
// .dataSync() converts TF tensors to plain JS Float32Arrays so we can
// accumulate them into regular arrays before creating the final 2D tensors.
function createTrainingData(context) {
  const inputs = []
  const outputs = []
  context.users
    // Only users with purchase history can provide meaningful training signal.
    // Users with no purchases have zero-ish vectors and no positive labels,
    // which would add noise without teaching the model anything useful.
    .filter((user) => user.purchases.length)
    .forEach((user) => {
      // Encode the user once, reuse for all product pairings.
      // .dataSync() extracts raw floats from the tensor into a typed array.
      const userVector = encodeUser(user, context).dataSync()
      context.products.forEach((product) => {
        // Encode each product
        const productVector = encodeProduct(product, context).dataSync()
        // Binary label: did this user purchase this specific product?
        const label = user.purchases.includes(product.id) ? 1 : 0
        // Concatenate user + product vectors into a single input row
        inputs.push([...userVector, ...productVector])
        outputs.push(label)
      })
    })

  return {
    // xs: 2D input tensor of shape [N*M, dimentions*2]
    xs: tf.tensor2d(inputs),
    // ys: 2D label tensor of shape [N*M, 1] — must be 2D for binaryCrossentropy
    ys: tf.tensor2d(outputs, [outputs.length, 1]),
    // inputDimensions: total width of each input row (user features + product features)
    inputDimensions: context.dimentions * 2
  }
}

// --------------------------------------------------------------------------
// NEURAL NETWORK — configureNeuralNetAndTrain()
// --------------------------------------------------------------------------

// Builds, compiles, and trains a feedforward neural network for binary classification.
//
// Architecture: 4 Dense layers forming a "funnel" that progressively compresses
// the concatenated [user, product] feature vector down to a single purchase probability.
//
//   Input (dimentions * 2) → Dense(128, relu) → Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)
//
// Why this architecture:
//   - 128→64→32 funnel: forces the network to learn increasingly abstract
//     representations. Early layers capture low-level feature interactions,
//     deeper layers learn high-level patterns like "users in this age group
//     prefer this type of product at this price range".
//   - ReLU activation: standard for hidden layers — fast to compute, avoids
//     vanishing gradient problem, introduces non-linearity.
//   - Sigmoid output: squashes the final value to [0, 1], interpretable as
//     the predicted probability that this user would buy this product.
//   - Binary cross-entropy loss: the standard loss function for binary
//     classification — measures how far the predicted probability is from
//     the actual 0/1 label.
//   - Adam optimizer (lr=0.01): adaptive learning rate optimizer that
//     converges faster than vanilla SGD. 0.01 is a moderately aggressive
//     learning rate suitable for small datasets.
//
// Returns the trained tf.Sequential model (NOT the fit() History object).
async function configureNeuralNetAndTrain(trainingData) {
  // tf.sequential() creates a linear stack of layers (no branching/merging)
  const model = tf.sequential()

  // Layer 1: input layer. inputShape must match the training data width.
  // 128 neurons give the network enough capacity to learn feature interactions.
  model.add(
    tf.layers.dense({
      units: 128,
      inputShape: [trainingData.inputDimensions],
      activation: 'relu'
    })
  )

  // Layer 2: compresses 128 → 64, forcing the network to distill patterns
  model.add(
    tf.layers.dense({
      units: 64,
      activation: 'relu'
    })
  )

  // Layer 3: compresses 64 → 32, further abstracting the representation
  model.add(
    tf.layers.dense({
      units: 32,
      activation: 'relu'
    })
  )

  // Layer 4: output layer — single neuron with sigmoid activation.
  // Outputs a value in [0, 1] representing P(user buys product).
  model.add(
    tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    })
  )

  // Compile: configure the optimizer, loss function, and metrics before training.
  // Must be called before model.fit().
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  })

  // Train the model on the full dataset.
  // - epochs: 100 full passes over the training data
  // - batchSize: 32 examples per gradient update (standard default)
  // - shuffle: true randomizes example order each epoch to prevent the model
  //   from learning the order of the data instead of the patterns
  // - onEpochEnd: posts loss and accuracy to the main thread after each epoch
  //   so the TFVisorView can render live training charts
  //   NOTE: TF.js v4.x uses `logs.acc` (not `logs.accuracy`) for the accuracy metric
  await model.fit(trainingData.xs, trainingData.ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        postMessage({
          type: workerEvents.trainingLog,
          epoch: epoch,
          loss: logs.loss,
          accuracy: logs.acc
        })
      }
    }
  })
  // Return the trained model itself — NOT the History object from model.fit().
  // The model is stored in _model for later use by recommend().
  return model
}

// --------------------------------------------------------------------------
// PIPELINE ORCHESTRATOR — trainModel()
// --------------------------------------------------------------------------

// Main training entry point — called when the worker receives a 'train:model' message.
// Orchestrates the full pipeline: context → encode → train → store.
//
// Receives { users, products } from WorkerController. Users have normalized
// purchases (arrays of product IDs like [1, 3, 7], not full product objects).
async function trainModel({ users, products }) {
  // Temporarily store products so they're accessible if recommend() is called
  // before training finishes (defensive — shouldn't happen in normal flow)
  _globalCtx.products = products
  console.log('Training model with users:', users)

  // Signal 50% progress to the UI (context building + encoding phase starting)
  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } })

  // Step 1: Build the context — normalization bounds, lookup indices, per-product stats
  const context = makeContext({ users, products })

  // Step 2: Pre-encode all product vectors and store them in the context.
  // These are reused during inference (recommend) to avoid re-encoding every product
  // each time a recommendation is requested. Each entry stores:
  //   - name: for display/debugging
  //   - meta: full product object (passed back to the UI in recommendations)
  //   - vector: Float32Array of the encoded features (from .dataSync())
  context.productVectors = products.map((product) => ({
    name: product.name,
    meta: { ...product },
    vector: encodeProduct(product, context).dataSync()
  }))

  // Replace _globalCtx with the fully built context (includes productVectors)
  _globalCtx = context

  // Step 3: Generate training data — all (user, product) pairs with binary labels
  const trainingData = createTrainingData(context)

  // Step 4: Build and train the neural network. Must happen AFTER createTrainingData
  // because the model's inputShape depends on trainingData.inputDimensions.
  // Stores the trained model in the module-level _model variable for inference.
  _model = await configureNeuralNetAndTrain(trainingData)

  // Signal 100% progress and training completion to the UI
  postMessage({
    type: workerEvents.progressUpdate,
    progress: { progress: 100 }
  })
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
