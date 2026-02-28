/**
 * UserService — data access layer for user data.
 *
 * Uses sessionStorage as a lightweight, ephemeral database:
 * - On first call to getDefaultUsers(), fetches users.json and seeds sessionStorage
 * - All subsequent reads/writes operate on sessionStorage (no server round-trips)
 * - Data is lost when the browser tab is closed (by design — it's a demo app)
 *
 * Methods are async to match the interface of a real API service,
 * making it easy to swap sessionStorage for a backend later.
 */
export class UserService {
  #storageKey = 'ew-academy-users'

  // Fetches the initial user data from the static JSON file
  // and seeds sessionStorage. This is the "database reset" — called on app startup.
  async getDefaultUsers() {
    const response = await fetch('./data/users.json')
    const users = await response.json()
    this.#setStorage(users)

    return users
  }

  async getUsers() {
    const users = this.#getStorage()
    return users
  }

  async getUserById(userId) {
    const users = this.#getStorage()
    return users.find((user) => user.id === userId)
  }

  // Merges updated fields into the existing user object and persists to sessionStorage
  async updateUser(user) {
    const users = this.#getStorage()
    const userIndex = users.findIndex((u) => u.id === user.id)

    users[userIndex] = { ...users[userIndex], ...user }
    this.#setStorage(users)

    return users[userIndex]
  }

  // Prepends a new user to the list (used for the "non-trained" test user)
  async addUser(user) {
    const users = this.#getStorage()
    this.#setStorage([user, ...users])
  }

  // Private helpers — sessionStorage get/set with JSON serialization
  #getStorage() {
    const data = sessionStorage.getItem(this.#storageKey)
    return data ? JSON.parse(data) : []
  }

  #setStorage(data) {
    sessionStorage.setItem(this.#storageKey, JSON.stringify(data))
  }
}
