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
    // New training started — clear previous epoch data so charts start fresh
    this.#events.onTrainModel(() => {
      this.#tfVisorView.resetDashboard()
    })

    // New epoch completed — forward the log to the visor for live chart updates
    this.#events.onTFVisLogs((log) => {
      this.#tfVisorView.handleTrainingLog(log)
    })
  }
}
