import 'dotenv/config'
import express from 'express'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import pg from 'pg'

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL })

const MODEL_DIR = path.join(process.cwd(), 'saved-model')

const app = express()
app.use(express.json({ limit: '50mb' }))
app.use(express.static('.'))

app.post('/api/model', async (req, res) => {
  const { modelTopology, weightSpecs, weightData } = req.body
  await fs.mkdir(MODEL_DIR, { recursive: true })

  const modelJson = {
    format: 'layers-model',
    modelTopology,
    weightsManifest: [{ paths: ['model.weights.bin'], weights: weightSpecs }]
  }
  await fs.writeFile(path.join(MODEL_DIR, 'model.json'), JSON.stringify(modelJson))
  await fs.writeFile(path.join(MODEL_DIR, 'model.weights.bin'), Buffer.from(weightData, 'base64'))

  res.status(201).end()
})

app.post('/api/model/context', async (req, res) => {
  await pool.query(
    `INSERT INTO model_snapshots (id, context, trained_at)
     VALUES (1, $1, NOW())
     ON CONFLICT (id) DO UPDATE SET context = EXCLUDED.context, trained_at = EXCLUDED.trained_at`,
    [req.body]
  )
  res.status(201).end()
})

app.get('/api/model/context', async (_req, res) => {
  const result = await pool.query('SELECT context FROM model_snapshots WHERE id = 1')
  if (result.rows.length === 0) return res.status(404).end()
  res.json(result.rows[0].context)
})

app.get('/api/model/:filename', (req, res) => {
  res.sendFile(path.join(MODEL_DIR, req.params.filename))
})

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
