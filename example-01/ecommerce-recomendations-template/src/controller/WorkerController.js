/**
 * WorkerController — the bridge between the Web Worker and the DOM event bus.
 *
 * Web Workers run on a separate thread and communicate via postMessage/onmessage.
 * The rest of the app uses DOM CustomEvents (the Events bus).
 * This controller translates between the two worlds:
 *
 * Main thread → Worker:
 *   Events.onTrainModel → worker.postMessage({ action: 'train:model', users })
 *   Events.onRecommend  → worker.postMessage({ action: 'recommend', user })
 *
 * Worker → Main thread:
 *   worker posts 'progress:update'    → Events.dispatchProgressUpdate()
 *   worker posts 'training:complete'  → Events.dispatchTrainingComplete()
 *   worker posts 'tfvis:data'         → Events.dispatchTFVisorData()
 *   worker posts 'training:log'       → Events.dispatchTFVisLogs()
 *   worker posts 'recommend'          → Events.dispatchRecommendationsReady()
 *
 * It also guards recommendations — only forwards `recommend` to the worker
 * if the model has already been trained (#alreadyTrained flag).
 */
import { workerEvents } from '../events/constants.js'

export class WorkerController {
  #worker
  #events
  #alreadyTrained = false
  constructor({ worker, events }) {
    this.#worker = worker
    this.#events = events
    this.#alreadyTrained = false
    this.init()
  }

  async init() {
    this.setupCallbacks()
  }

  static init(deps) {
    return new WorkerController(deps)
  }

  setupCallbacks() {
    // Main thread → Worker: forward training requests
    this.#events.onTrainModel((data) => {
      this.#alreadyTrained = false
      this.triggerTrain(data)
    })
    this.#events.onTrainingComplete(() => {
      this.#alreadyTrained = true
    })

    // Main thread → Worker: forward recommendation requests (only if trained)
    this.#events.onRecommend((data) => {
      if (!this.#alreadyTrained) return

      this.triggerRecommend(data)
    })

    // Worker → Main thread: translate worker postMessage events into DOM CustomEvents.
    // Known event types are not logged to keep the console clean.
    const eventsToIgnoreLogs = [
      workerEvents.progressUpdate,
      workerEvents.trainingLog,
      workerEvents.tfVisData,
      workerEvents.tfVisLogs,
      workerEvents.trainingComplete
    ]
    this.#worker.onmessage = (event) => {
      if (!eventsToIgnoreLogs.includes(event.data.type)) console.log(event.data)

      if (event.data.type === workerEvents.progressUpdate) {
        this.#events.dispatchProgressUpdate(event.data.progress)
      }

      if (event.data.type === workerEvents.trainingComplete) {
        this.#events.dispatchTrainingComplete(event.data)
      }

      if (event.data.type === workerEvents.tfVisData) {
        this.#events.dispatchTFVisorData(event.data.data)
      }

      if (event.data.type === workerEvents.trainingLog) {
        this.#events.dispatchTFVisLogs(event.data)
      }

      if (event.data.type === workerEvents.recommend) {
        this.#events.dispatchRecommendationsReady(event.data)
      }
    }
  }

  // Sends a training request to the worker thread
  triggerTrain(users) {
    this.#worker.postMessage({ action: workerEvents.trainModel, users })
  }

  // Sends a recommendation request to the worker thread
  triggerRecommend(user) {
    this.#worker.postMessage({ action: workerEvents.recommend, user })
  }
}
