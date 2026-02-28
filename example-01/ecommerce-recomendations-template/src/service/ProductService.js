/**
 * ProductService — read-only data access for the product catalog.
 *
 * Unlike UserService, products are not mutable — they're always
 * fetched fresh from the static JSON file. This is the product "database".
 */
export class ProductService {
  async getProducts() {
    const response = await fetch('./data/products.json')
    return await response.json()
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
