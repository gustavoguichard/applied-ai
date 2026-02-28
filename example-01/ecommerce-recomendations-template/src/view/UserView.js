/**
 * UserView — manages the user profile section of the UI.
 *
 * Responsibilities:
 * - Renders user options in the <select> dropdown
 * - Displays selected user's age and past purchases
 * - Handles purchase removal (click to remove with fade-out animation)
 * - Handles adding new purchases with a highlight animation
 *
 * Like all views, it delegates business logic to the controller via callbacks:
 *   - registerUserSelectCallback → called when user picks a user from the dropdown
 *   - registerPurchaseRemoveCallback → called when user clicks a past purchase to remove it
 */
import { View } from './View.js'

export class UserView extends View {
  // DOM element references — grabbed once at construction time
  #userSelect = document.querySelector('#userSelect')
  #userAge = document.querySelector('#userAge')
  #pastPurchasesList = document.querySelector('#pastPurchasesList')

  #purchaseTemplate
  // Callbacks registered by the controller (Inversion of Control)
  #onUserSelect
  #onPurchaseRemove
  #pastPurchaseElements = []

  constructor() {
    super()
    this.init()
  }

  async init() {
    this.#purchaseTemplate = await this.loadTemplate(
      './src/view/templates/past-purchase.html'
    )
    this.attachUserSelectListener()
  }

  registerUserSelectCallback(callback) {
    this.#onUserSelect = callback
  }

  registerPurchaseRemoveCallback(callback) {
    this.#onPurchaseRemove = callback
  }

  // Appends <option> elements for each user into the dropdown
  renderUserOptions(users) {
    const options = users
      .map((user) => {
        return `<option value="${user.id}">${user.name}</option>`
      })
      .join('')

    this.#userSelect.innerHTML += options
  }

  renderUserDetails(user) {
    this.#userAge.value = user.age
  }

  // Renders the full list of past purchases for the currently selected user.
  // Each purchase is rendered from the past-purchase.html template.
  renderPastPurchases(pastPurchases) {
    if (!this.#purchaseTemplate) return

    if (!pastPurchases || pastPurchases.length === 0) {
      this.#pastPurchasesList.innerHTML = '<p>No past purchases found.</p>'
      return
    }

    const html = pastPurchases
      .map((product) => {
        return this.replaceTemplate(this.#purchaseTemplate, {
          ...product,
          product: JSON.stringify(product)
        })
      })
      .join('')

    this.#pastPurchasesList.innerHTML = html
    this.attachPurchaseClickHandlers()
  }

  // Adds a single new purchase to the top of the list with a green highlight animation.
  // Called by the controller after a "Buy Now" action succeeds.
  addPastPurchase(product) {
    if (this.#pastPurchasesList.innerHTML.includes('No past purchases found')) {
      this.#pastPurchasesList.innerHTML = ''
    }

    const purchaseHtml = this.replaceTemplate(this.#purchaseTemplate, {
      ...product,
      product: JSON.stringify(product)
    })

    this.#pastPurchasesList.insertAdjacentHTML('afterbegin', purchaseHtml)

    const newPurchase =
      this.#pastPurchasesList.firstElementChild.querySelector('.past-purchase')
    newPurchase.classList.add('past-purchase-highlight')

    setTimeout(() => {
      newPurchase.classList.remove('past-purchase-highlight')
    }, 1000)

    this.attachPurchaseClickHandlers()
  }

  // Listens for changes on the user <select> dropdown.
  // Delegates to the controller callback, or clears the UI if "-- Select a user --" is chosen.
  attachUserSelectListener() {
    this.#userSelect.addEventListener('change', (event) => {
      const userId = event.target.value ? Number(event.target.value) : null

      if (userId) {
        if (this.#onUserSelect) {
          this.#onUserSelect(userId)
        }
      } else {
        this.#userAge.value = ''
        this.#pastPurchasesList.innerHTML = ''
      }
    })
  }

  // Makes each past purchase clickable for removal.
  // On click: notifies the controller, fades out the element, then removes it from the DOM.
  attachPurchaseClickHandlers() {
    this.#pastPurchaseElements = []

    const purchaseElements = document.querySelectorAll('.past-purchase')

    purchaseElements.forEach((purchaseElement) => {
      this.#pastPurchaseElements.push(purchaseElement)

      purchaseElement.onclick = () => {
        const product = JSON.parse(purchaseElement.dataset.product)
        const userId = this.getSelectedUserId()
        const element = purchaseElement.closest('.col-md-6')

        this.#onPurchaseRemove({ element, userId, product })

        element.style.transition = 'opacity 0.5s ease'
        element.style.opacity = '0'

        setTimeout(() => {
          element.remove()

          if (document.querySelectorAll('.past-purchase').length === 0) {
            this.renderPastPurchases([])
          }
        }, 500)
      }
    })
  }

  getSelectedUserId() {
    return this.#userSelect.value ? Number(this.#userSelect.value) : null
  }
}
