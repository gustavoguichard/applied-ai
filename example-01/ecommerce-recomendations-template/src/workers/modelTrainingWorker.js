/**
 * Model Training Web Worker — runs TensorFlow.js model training off the main thread.
 *
 * Web Workers run in a separate thread, so heavy ML computations
 * don't block the UI. Communication with the main thread is via:
 *   - self.onmessage: receives commands from WorkerController
 *   - postMessage(): sends results/progress back to WorkerController
 *
 * Message protocol (matches workerEvents in constants.js):
 *   Inbound:  { action: 'train:model', users: [...] }
 *   Inbound:  { action: 'recommend', user: {...} }
 *   Outbound: { type: 'progress:update', progress: { progress: 50 } }
 *   Outbound: { type: 'training:log', epoch, loss, accuracy }
 *   Outbound: { type: 'training:complete' }
 *   Outbound: { type: 'recommend', user, recommendations: [...] }
 *
 * CURRENT STATE: This is a stub/template. The actual TensorFlow.js model
 * training and inference logic needs to be implemented.
 * - trainModel() fakes 50% → 100% progress with a 1s delay
 * - recommend() logs to console but doesn't return recommendations yet
 *
 * _globalCtx is intended to hold the trained model and any shared state
 * (weights, catalog mapping, etc.) between train and recommend calls.
 */
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js'
import { workerEvents } from '../events/constants.js'

console.log('Model training worker initialized')

// Shared state between trainModel and recommend — will hold the trained model
const _globalCtx = {}

// Stub: replace with actual TensorFlow.js model.fit() logic
async function trainModel({ users }) {
  console.log('Training model with users:', users)

  postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } })
  postMessage({
    type: workerEvents.trainingLog,
    epoch: 1,
    loss: 1,
    accuracy: 1
  })

  setTimeout(() => {
    postMessage({
      type: workerEvents.progressUpdate,
      progress: { progress: 100 }
    })
    postMessage({ type: workerEvents.trainingComplete })
  }, 1000)
}

// Stub: replace with actual model.predict() logic
function recommend(user, _ctx) {
  console.log('will recommend for user:', user)
  // postMessage({
  //     type: workerEvents.recommend,
  //     user,
  //     recommendations: []
  // });
}

// Action router — maps incoming action names to handler functions
const handlers = {
  [workerEvents.trainModel]: trainModel,
  [workerEvents.recommend]: (d) => recommend(d.user, _globalCtx)
}

// Entry point for all messages from WorkerController.
// Destructures { action, ...data } and routes to the appropriate handler.
self.onmessage = (e) => {
  const { action, ...data } = e.data
  if (handlers[action]) handlers[action](data)
}
