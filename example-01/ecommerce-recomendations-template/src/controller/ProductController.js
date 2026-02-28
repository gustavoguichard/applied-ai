/**
 * ProductController — connects ProductView with services and the event bus.
 *
 * Lifecycle:
 * 1. On init: fetches all products → renders cards (buttons disabled)
 * 2. Listens for `user:selected` → enables buy buttons + requests recommendations
 * 3. Listens for `recommendations:ready` → re-renders product list with recommended items
 * 4. On "Buy Now" click → dispatches `purchase:added` (UserController picks this up)
 *
 * This controller does NOT directly talk to UserController.
 * All communication goes through the Events bus.
 */
export class ProductController {
  #productView
  #currentUser = null
  #events
  #productService
  constructor({ productView, events, productService }) {
    this.#productView = productView
    this.#productService = productService
    this.#events = events
    this.init()
  }

  static init(deps) {
    return new ProductController(deps)
  }

  async init() {
    this.setupCallbacks()
    this.setupEventListeners()
    const products = await this.#productService.getProducts()
    this.#productView.render(products, true)
  }

  setupEventListeners() {
    // When a user is selected, enable buy buttons and ask the worker for recommendations
    this.#events.onUserSelected((user) => {
      this.#currentUser = user
      this.#productView.onUserSelected(user)
      this.#events.dispatchRecommend(user)
    })

    // When the worker returns recommendations, re-render the product list with them
    this.#events.onRecommendationsReady(({ recommendations }) => {
      this.#productView.render(recommendations, false)
    })
  }

  // Registers the "Buy Now" handler on the view (Inversion of Control)
  setupCallbacks() {
    this.#productView.registerBuyProductCallback(
      this.handleBuyProduct.bind(this)
    )
  }

  // When a product is bought, broadcast it so UserController can update the user's purchases
  async handleBuyProduct(product) {
    const user = this.#currentUser
    this.#events.dispatchPurchaseAdded({ user, product })
  }
}
