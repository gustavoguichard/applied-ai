/**
 * UserController — manages the user lifecycle: selection, purchases, and profile display.
 *
 * Responsibilities:
 * - Renders user options in the dropdown (default users + the non-trained test user)
 * - Handles user selection → dispatches `user:selected` so other controllers react
 * - Listens for `purchase:added` (from ProductController) → updates user data in storage
 * - Handles purchase removal → updates storage and dispatches `users:updated`
 *
 * The `users:updated` event is important — ModelController listens for it
 * to refresh the "All Users Purchase Data" display.
 */
export class UserController {
  #userService
  #productService
  #userView
  #events
  constructor({ userView, userService, productService, events }) {
    this.#userView = userView
    this.#userService = userService
    this.#productService = productService
    this.#events = events
  }

  static init(deps) {
    return new UserController(deps)
  }

  // Called from index.js with the non-trained user.
  // Loads default users, adds the test user, renders the dropdown,
  // and notifies the system that the users list is ready.
  async renderUsers(nonTrainedUser) {
    const users = await this.#userService.getDefaultUsers()

    this.#userService.addUser(nonTrainedUser)
    const defaultAndNonTrained = [nonTrainedUser, ...users]

    this.#userView.renderUserOptions(defaultAndNonTrained)
    this.setupCallbacks()
    this.setupPurchaseObserver()

    this.#events.dispatchUsersUpdated({ users: defaultAndNonTrained })
  }

  // Registers view callbacks — the view calls these when the user interacts with the UI
  setupCallbacks() {
    this.#userView.registerUserSelectCallback(this.handleUserSelect.bind(this))
    this.#userView.registerPurchaseRemoveCallback(
      this.handlePurchaseRemove.bind(this)
    )
  }

  // Subscribes to `purchase:added` events from ProductController.
  // This is how the "Buy Now" action in the product catalog
  // ends up updating the user's purchase history.
  setupPurchaseObserver() {
    this.#events.onPurchaseAdded(async (...data) => {
      return this.handlePurchaseAdded(...data)
    })
  }

  // User selected from dropdown → fetch their data, broadcast the selection,
  // and display their profile details + purchase history.
  async handleUserSelect(userId) {
    const user = await this.#userService.getUserById(userId)
    this.#events.dispatchUserSelected(user)
    return this.displayUserDetails(user)
  }

  // A product was bought → insert a row into the purchases table,
  // update the UI, and notify other controllers.
  async handlePurchaseAdded({ user, product }) {
    await this.#userService.addPurchase(user.id, product.id)

    this.#userView.addPastPurchase(product)
    this.#events.dispatchUsersUpdated({
      users: await this.#userService.getUsers()
    })
    const updatedUser = await this.#userService.getUserById(user.id)
    this.#events.dispatchRecommend(updatedUser)
  }

  // A past purchase was clicked for removal → delete the row from the purchases table
  // and notify other controllers to refresh their displays.
  async handlePurchaseRemove({ userId, product }) {
    await this.#userService.removePurchase(userId, product.id)

    const updatedUser = await this.#userService.getUserById(userId)
    const updatedUsers = await this.#userService.getUsers()
    this.#events.dispatchUsersUpdated({ users: updatedUsers })
    this.#events.dispatchRecommend(updatedUser)
  }

  // Resolve purchase IDs to full product objects before passing to the view.
  async displayUserDetails(user) {
    this.#userView.renderUserDetails(user)
    const products = await this.#productService.getProductsByIds(user.purchases)
    this.#userView.renderPastPurchases(products)
  }

  getSelectedUserId() {
    return this.#userView.getSelectedUserId()
  }
}
