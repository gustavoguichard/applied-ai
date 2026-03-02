import 'dotenv/config'
import { readFileSync } from 'node:fs'
import pg from 'pg'

const { Pool } = pg

const pool = new Pool({ connectionString: process.env.DATABASE_URL })

const products = JSON.parse(readFileSync('./data/products.json', 'utf-8'))
const users = JSON.parse(readFileSync('./data/users.json', 'utf-8'))

async function seed() {
  const client = await pool.connect()

  try {
    await client.query('BEGIN')

    await client.query('TRUNCATE purchases, users, products RESTART IDENTITY CASCADE')

    for (const product of products) {
      await client.query(
        'INSERT INTO products (id, name, category, price, color) VALUES ($1, $2, $3, $4, $5)',
        [product.id, product.name, product.category, product.price, product.color]
      )
    }

    for (const user of users) {
      await client.query(
        'INSERT INTO users (id, name, age) VALUES ($1, $2, $3)',
        [user.id, user.name, user.age]
      )

      for (const productId of user.purchases) {
        await client.query(
          'INSERT INTO purchases (user_id, product_id) VALUES ($1, $2)',
          [user.id, productId]
        )
      }
    }

    await client.query(
      "SELECT setval('products_id_seq', (SELECT MAX(id) FROM products))"
    )

    await client.query('COMMIT')
    console.log('Seed complete.')
  } catch (err) {
    await client.query('ROLLBACK')
    console.error('Seed failed:', err)
    process.exit(1)
  } finally {
    client.release()
    await pool.end()
  }
}

seed()
