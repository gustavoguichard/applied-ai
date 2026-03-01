/**
 * TFVisorController — connects the event bus to the TFVisorView (training charts).
 *
 * Simple passthrough controller:
 * - On `training:train` → resets the dashboard (clears old chart data)
 * - On `tfvis:logs` → feeds each training epoch's data to the visor for charting
 */
export class TFVisorController {
  #tfVisorView
  #events
  constructor({ tfVisorView, events }) {
    this.#tfVisorView = tfVisorView
    this.#events = events

    this.init()
  }

  static init(deps) {
    return new TFVisorController(deps)
  }

  async init() {
    this.setupCallbacks()
  }

  setupCallbacks() {
    this.#events.onTrainModel(() => {
      this.#tfVisorView.resetDashboard()
      this.#tfVisorView.open()
    })

    this.#events.onTFVisLogs((log) => {
      this.#tfVisorView.handleTrainingLog(log)
    })

    this.#events.onTrainingComplete(() => {
      setTimeout(() => this.#tfVisorView.close(), 2000)
    })
  }
}
