/**
 * ProductView — renders the product catalog cards and handles "Buy Now" button interactions.
 *
 * Responsibilities:
 * - Loads the product-card.html template on construction
 * - Renders a list of products as Bootstrap cards
 * - Manages button state (disabled until a user is selected)
 * - Provides visual feedback on purchase (brief green "Added" flash)
 *
 * The view does NOT know about services or events — it delegates business logic
 * to the controller via the registered callback (Inversion of Control pattern):
 *   controller calls → view.registerBuyProductCallback(handler)
 *   user clicks "Buy Now" → view calls handler(product)
 */
import { View } from './View.js'

export class ProductView extends View {
  #productList = document.querySelector('#productList')

  #buttons
  #productTemplate
  // Callback provided by the controller — called when user clicks "Buy Now"
  #onBuyProduct

  constructor() {
    super()
    this.init()
  }

  async init() {
    this.#productTemplate = await this.loadTemplate(
      './src/view/templates/product-card.html'
    )
  }

  // Called by the controller when a user is selected/deselected.
  // Enables buy buttons only when there's a valid user (so purchases can be attributed).
  onUserSelected(user) {
    this.setButtonsState(!user.id)
  }

  // Inversion of Control: the controller registers what should happen on "Buy Now".
  registerBuyProductCallback(callback) {
    this.#onBuyProduct = callback
  }

  // Renders all product cards into #productList.
  // Each product's full JSON is embedded in data-product for easy retrieval on click.
  render(products, disableButtons = true) {
    if (!this.#productTemplate) return
    const html = products
      .map((product) => {
        return this.replaceTemplate(this.#productTemplate, {
          id: product.id,
          name: product.name,
          category: product.category,
          price: product.price,
          color: product.color,
          product: JSON.stringify(product)
        })
      })
      .join('')

    this.#productList.innerHTML = html
    this.attachBuyButtonListeners()

    this.setButtonsState(disableButtons)
  }

  setButtonsState(disabled) {
    if (!this.#buttons) {
      this.#buttons = document.querySelectorAll('.buy-now-btn')
    }
    this.#buttons.forEach((button) => {
      button.disabled = disabled
    })
  }

  // Attaches click listeners to all "Buy Now" buttons.
  // On click: shows a brief green confirmation, then delegates to the controller callback.
  attachBuyButtonListeners() {
    this.#buttons = document.querySelectorAll('.buy-now-btn')
    this.#buttons.forEach((button) => {
      button.addEventListener('click', () => {
        const product = JSON.parse(button.dataset.product)
        const originalText = button.innerHTML

        button.innerHTML = '<i class="bi bi-check-circle-fill"></i> Added'
        button.classList.remove('btn-primary')
        button.classList.add('btn-success')
        setTimeout(() => {
          button.innerHTML = originalText
          button.classList.remove('btn-success')
          button.classList.add('btn-primary')
        }, 500)
        this.#onBuyProduct(product, button)
      })
    })
  }
}
