import { makeService } from 'make-service'

export const api = makeService('/api', {
  headers: { 'Content-Type': 'application/json' },
})
