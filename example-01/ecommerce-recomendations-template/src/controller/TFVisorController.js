/**
 * TFVisorController — connects the event bus to the TFVisorView (training charts).
 *
 * Simple passthrough controller:
 * - On `training:train`    → resets the dashboard (clears old chart data)
 * - On `training:log`      → feeds each training epoch's data to the visor for charting
 * - On `training:complete` → schedules the visor to close after 2 seconds
 */
export class TFVisorController {
  #tfVisorView
  #events
  #closeTimeout = null
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
      // Cancel any pending auto-close from a previous training or page-load restore,
      // so re-training always gets a fresh visor that stays open for the full run.
      clearTimeout(this.#closeTimeout)
      this.#tfVisorView.resetDashboard()
      this.#tfVisorView.open()
    })

    this.#events.onTFVisLogs((log) => {
      this.#tfVisorView.handleTrainingLog(log)
    })

    this.#events.onTrainingComplete(() => {
      this.#closeTimeout = setTimeout(() => this.#tfVisorView.close(), 2000)
    })
  }
}
