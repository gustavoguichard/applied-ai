import { z } from 'zod'
import { api } from './api.js'

const productSchema = z.object({
  id: z.number(),
  name: z.string(),
  category: z.string(),
  price: z.coerce.number(),
  color: z.string(),
})

const productsSchema = z.array(productSchema)

export class ProductService {
  #products = null

  async getProducts() {
    if (!this.#products) {
      const response = await api.get('/products')
      this.#products = await response.json(productsSchema)
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
