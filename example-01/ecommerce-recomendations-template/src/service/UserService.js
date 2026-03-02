/**
 * UserService — data access layer for user data.
 *
 * Fetches user data from the REST API backed by PostgreSQL.
 * Purchases are tracked in a dedicated `purchases` table and
 * returned as an array of product IDs on each user object.
 *
 * Methods are async to match a real API service interface.
 */
export class UserService {
  // Alias for getUsers() — kept for interface compatibility with existing callers.
  // The DB is already seeded, so there is no separate "default" state to load.
  async getDefaultUsers() {
    return this.getUsers()
  }

  async getUsers() {
    const response = await fetch('/api/users')
    return response.json()
  }

  async getUserById(userId) {
    const response = await fetch(`/api/users/${userId}`)
    return response.json()
  }

  // Creates a new user. Uses an upsert on the server so re-loading the page
  // with a hardcoded user (e.g. the non-trained test user) is idempotent.
  async addUser(user) {
    await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(user),
    })
  }

  // Inserts a row into the `purchases` table. Idempotent — duplicate buys are ignored.
  async addPurchase(userId, productId) {
    await fetch('/api/purchases', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, product_id: productId }),
    })
  }

  // Removes a row from the `purchases` table.
  async removePurchase(userId, productId) {
    await fetch(`/api/purchases/${userId}/${productId}`, { method: 'DELETE' })
  }
}
