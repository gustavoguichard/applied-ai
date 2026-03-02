/**
 * ProductService — read-only data access for the product catalog.
 *
 * Fetches products from the REST API backed by PostgreSQL.
 * The catalog is cached after the first fetch since it never changes at runtime.
 */
export class ProductService {
  #products = null

  async getProducts() {
    if (!this.#products) {
      const response = await fetch('/api/products')
      this.#products = await response.json()
    }
    return this.#products
  }

  async getProductById(id) {
    const products = await this.getProducts()
    return products.find((product) => product.id === id)
  }

  async getProductsByIds(ids) {
    const products = await this.getProducts()
    return products.filter((product) => ids.includes(product.id))
  }
}
