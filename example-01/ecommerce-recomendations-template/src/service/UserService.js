import { z } from 'zod'
import { api } from './api.js'

const userSchema = z.object({
  id: z.number(),
  name: z.string(),
  age: z.number(),
  purchases: z.array(z.number()),
})

const usersSchema = z.array(userSchema)

export class UserService {
  async getDefaultUsers() {
    return this.getUsers()
  }

  async getUsers() {
    const response = await api.get('/users')
    return response.json(usersSchema)
  }

  async getUserById(userId) {
    const response = await api.get('/users/:id', { params: { id: String(userId) } })
    return response.json(userSchema)
  }

  async addUser(user) {
    await api.post('/users', { body: user })
  }

  async addPurchase(userId, productId) {
    await api.post('/purchases', { body: { user_id: userId, product_id: productId } })
  }

  async removePurchase(userId, productId) {
    await api.delete('/purchases/:userId/:productId', {
      params: { userId: String(userId), productId: String(productId) },
    })
  }
}
