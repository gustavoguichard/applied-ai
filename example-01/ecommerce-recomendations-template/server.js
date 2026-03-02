import 'dotenv/config'
import express from 'express'
import pg from 'pg'

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL })

const app = express()
app.use(express.json())
app.use(express.static('.'))

app.get('/api/products', async (_req, res) => {
  const result = await pool.query('SELECT * FROM products ORDER BY id')
  res.json(result.rows)
})

app.get('/api/users', async (_req, res) => {
  const result = await pool.query(`
    SELECT u.id, u.name, u.age,
      COALESCE(
        array_agg(p.product_id ORDER BY p.product_id) FILTER (WHERE p.product_id IS NOT NULL),
        '{}'
      ) AS purchases
    FROM users u
    LEFT JOIN purchases p ON p.user_id = u.id
    GROUP BY u.id
    ORDER BY u.id
  `)
  res.json(result.rows)
})

app.get('/api/users/:id', async (req, res) => {
  const result = await pool.query(
    `
    SELECT u.id, u.name, u.age,
      COALESCE(
        array_agg(p.product_id ORDER BY p.product_id) FILTER (WHERE p.product_id IS NOT NULL),
        '{}'
      ) AS purchases
    FROM users u
    LEFT JOIN purchases p ON p.user_id = u.id
    WHERE u.id = $1
    GROUP BY u.id
    `,
    [req.params.id]
  )

  if (result.rows.length === 0) return res.status(404).json({ error: 'User not found' })
  res.json(result.rows[0])
})

app.post('/api/users', async (req, res) => {
  const { id, name, age } = req.body
  await pool.query(
    'INSERT INTO users (id, name, age) VALUES ($1, $2, $3) ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, age = EXCLUDED.age',
    [id, name, age]
  )
  res.status(201).json({ id, name, age, purchases: [] })
})

app.post('/api/purchases', async (req, res) => {
  const { user_id, product_id } = req.body
  await pool.query(
    'INSERT INTO purchases (user_id, product_id) VALUES ($1, $2) ON CONFLICT DO NOTHING',
    [user_id, product_id]
  )
  res.status(201).end()
})

app.delete('/api/purchases/:userId/:productId', async (req, res) => {
  await pool.query(
    'DELETE FROM purchases WHERE user_id = $1 AND product_id = $2',
    [req.params.userId, req.params.productId]
  )
  res.status(204).end()
})

const PORT = process.env.PORT ?? 3000
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`))
