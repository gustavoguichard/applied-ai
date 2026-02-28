/**
 * ProductService — read-only data access for the product catalog.
 *
 * Unlike UserService, products are not mutable — they're always
 * fetched from the static JSON file. The catalog is cached after
 * the first fetch since it never changes at runtime.
 */
export class ProductService {
  #products = null

  async getProducts() {
    if (!this.#products) {
      const response = await fetch('./data/products.json')
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
