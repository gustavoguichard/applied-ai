/**
 * ModelView — manages the "Model Training" card in the UI.
 *
 * Responsibilities:
 * - "Train Model" button → triggers model training via controller callback
 * - "Run Recommendation" button → triggers recommendation via controller callback
 * - Collapsible "All Users Purchase Data" section (toggle show/hide)
 * - Training progress UI (spinner + disabled state during training)
 *
 * The "Run Recommendation" button starts disabled and is only enabled
 * when BOTH conditions are met: a model has been trained AND a user is selected.
 */
import { View } from './View.js'

export class ModelView extends View {
  #trainModelBtn = document.querySelector('#trainModelBtn')
  #purchasesArrow = document.querySelector('#purchasesArrow')
  #purchasesDiv = document.querySelector('#purchasesDiv')
  #allUsersPurchasesList = document.querySelector('#allUsersPurchasesList')
  #runRecommendationBtn = document.querySelector('#runRecommendationBtn')
  // Callbacks registered by ModelController
  #onTrainModel
  #onRunRecommendation

  constructor() {
    super()
    this.attachEventListeners()
  }

  registerTrainModelCallback(callback) {
    this.#onTrainModel = callback
  }
  registerRunRecommendationCallback(callback) {
    this.#onRunRecommendation = callback
  }

  attachEventListeners() {
    // Delegate button clicks to the controller
    this.#trainModelBtn.addEventListener('click', () => {
      this.#onTrainModel()
    })
    this.#runRecommendationBtn.addEventListener('click', () => {
      this.#onRunRecommendation()
    })

    // Toggle the collapsible "All Users Purchase Data" section
    this.#purchasesDiv.addEventListener('click', () => {
      const purchasesList = this.#allUsersPurchasesList

      const isHidden = window.getComputedStyle(purchasesList).display === 'none'

      if (isHidden) {
        purchasesList.style.display = 'block'
        this.#purchasesArrow.classList.remove('bi-chevron-down')
        this.#purchasesArrow.classList.add('bi-chevron-up')
      } else {
        purchasesList.style.display = 'none'
        this.#purchasesArrow.classList.remove('bi-chevron-up')
        this.#purchasesArrow.classList.add('bi-chevron-down')
      }
    })
  }
  enableRecommendButton() {
    this.#runRecommendationBtn.disabled = false
  }

  // Updates the train button to show a spinner during training,
  // and re-enables it when progress reaches 100%.
  updateTrainingProgress(progress) {
    this.#trainModelBtn.disabled = true
    this.#trainModelBtn.innerHTML =
      '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...'

    if (progress.progress === 100) {
      this.#trainModelBtn.disabled = false
      this.#trainModelBtn.innerHTML = 'Train Recommendation Model'
    }
  }

  // Renders a summary of all users and their purchases (badges) in the collapsible section.
  // This gives visibility into the training data used by the model.
  renderAllUsersPurchases(users) {
    const html = users
      .map((user) => {
        const purchasesHtml = user.purchases
          .map((purchase) => {
            return `<span class="badge bg-light text-dark me-1 mb-1">${purchase.name}</span>`
          })
          .join('')

        return `
                <div class="user-purchase-summary">
                    <h6>${user.name} (Age: ${user.age})</h6>
                    <div class="purchases-badges">
                        ${purchasesHtml || '<span class="text-muted">No purchases</span>'}
                    </div>
                </div>
            `
      })
      .join('')

    this.#allUsersPurchasesList.innerHTML = html
  }
}
