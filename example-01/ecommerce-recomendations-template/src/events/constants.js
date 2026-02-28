/**
 * Event name constants split into two namespaces:
 *
 * `events` — used on the main thread (DOM CustomEvents dispatched on `document`).
 *   Controllers subscribe/publish through the Events class using these names.
 *
 * `workerEvents` — used for Web Worker communication (postMessage / onmessage).
 *   The WorkerController translates between these two namespaces,
 *   acting as the bridge between the worker thread and the DOM event bus.
 */

// Main-thread event names (DOM CustomEvents)
export const events = {
  userSelected: 'user:selected',
  usersUpdated: 'users:updated',
  purchaseAdded: 'purchase:added',
  purchaseRemoved: 'purchase:remove',
  modelTrain: 'training:train',
  trainingComplete: 'training:complete',
  modelProgressUpdate: 'model:progress-update',
  recommendationsReady: 'recommendations:ready',
  recommend: 'recommend'
}

// Worker-thread event names (postMessage protocol)
export const workerEvents = {
  trainingComplete: 'training:complete',
  trainModel: 'train:model',
  recommend: 'recommend',
  trainingLog: 'training:log',
  progressUpdate: 'progress:update',
  tfVisData: 'tfvis:data',
  tfVisLogs: 'tfvis:logs'
}
