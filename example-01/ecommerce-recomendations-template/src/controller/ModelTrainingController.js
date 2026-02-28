/**
 * ModelController — orchestrates model training and recommendation requests.
 *
 * This controller manages the state machine for the ML workflow:
 *
 *   [Not Trained] → click "Train Model" → dispatches `training:train` → [Training...]
 *   [Training...] → worker sends progress → updates spinner/button
 *   [Training...] → worker sends `training:complete` → [Trained]
 *   [Trained] + [User Selected] → enables "Run Recommendation" button
 *   click "Run Recommendation" → dispatches `recommend` → worker runs inference
 *
 * The "Run Recommendation" button is only enabled when both conditions are true:
 *   1. #alreadyTrained === true (model has been trained at least once)
 *   2. #currentUser !== null (a user is selected from the dropdown)
 */
export class ModelController {
  #modelView
  #userService
  #events
  #currentUser = null
  #alreadyTrained = false
  constructor({ modelView, userService, events }) {
    this.#modelView = modelView
    this.#userService = userService
    this.#events = events

    this.init()
  }

  static init(deps) {
    return new ModelController(deps)
  }

  async init() {
    this.setupCallbacks()
  }

  setupCallbacks() {
    // Register button click handlers on the view
    this.#modelView.registerTrainModelCallback(this.handleTrainModel.bind(this))
    this.#modelView.registerRunRecommendationCallback(
      this.handleRunRecommendation.bind(this)
    )

    // Track the currently selected user — enable recommend button if model is ready
    this.#events.onUserSelected((user) => {
      this.#currentUser = user
      if (!this.#alreadyTrained) return
      this.#modelView.enableRecommendButton()
    })

    // Track training completion — enable recommend button if a user is selected
    this.#events.onTrainingComplete(() => {
      this.#alreadyTrained = true
      if (!this.#currentUser) return
      this.#modelView.enableRecommendButton()
    })

    // When the user list changes (purchase added/removed), refresh the summary display
    this.#events.onUsersUpdated(async (...data) => {
      return this.refreshUsersPurchaseData(...data)
    })

    // Update the training button UI (spinner, progress) during training
    this.#events.onProgressUpdate((progress) => {
      this.handleTrainingProgressUpdate(progress)
    })
  }

  // "Train Model" clicked → get all users and dispatch to the worker via event bus
  async handleTrainModel() {
    const users = await this.#userService.getUsers()

    this.#events.dispatchTrainModel(users)
  }

  handleTrainingProgressUpdate(progress) {
    this.#modelView.updateTrainingProgress(progress)
  }

  // "Run Recommendation" clicked → get the latest user data and request recommendations
  async handleRunRecommendation() {
    const currentUser = this.#currentUser
    const updatedUser = await this.#userService.getUserById(currentUser.id)
    this.#events.dispatchRecommend(updatedUser)
  }

  async refreshUsersPurchaseData({ users }) {
    this.#modelView.renderAllUsersPurchases(users)
  }
}
