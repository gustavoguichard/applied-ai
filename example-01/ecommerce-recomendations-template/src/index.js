/**
 * Application Entry Point (Bootstrapper)
 *
 * This file wires together the entire application using manual dependency injection.
 * There is no framework — each layer is instantiated here and connected explicitly:
 *
 * 1. Services are created (data layer)
 * 2. Views are created (DOM layer — they grab their DOM elements on construction)
 * 3. A Web Worker is spawned for off-thread ML training
 * 4. Controllers are created, receiving their dependencies (views, services, event bus)
 * 5. Initial data is loaded and the first training cycle is triggered
 *
 * Controllers never know about each other. They communicate exclusively
 * through the Events bus (pub/sub via DOM CustomEvents).
 */
import { ModelController } from './controller/ModelTrainingController.js'
import { ProductController } from './controller/ProductController.js'
import { TFVisorController } from './controller/TFVisorController.js'
import { UserController } from './controller/UserController.js'
import { WorkerController } from './controller/WorkerController.js'
import Events from './events/events.js'
import { ProductService } from './service/ProductService.js'
import { UserService } from './service/UserService.js'
import { ModelView } from './view/ModelTrainingView.js'
import { ProductView } from './view/ProductView.js'
import { TFVisorView } from './view/TFVisorView.js'
import { UserView } from './view/UserView.js'

// --- Data layer ---
const userService = new UserService()
const productService = new ProductService()

// --- DOM layer ---
// Each view grabs its DOM elements via querySelector on construction
const userView = new UserView()
const productView = new ProductView()
const modelView = new ModelView()
const tfVisorView = new TFVisorView()

// --- Web Worker for ML ---
// Runs TensorFlow.js model training on a separate thread to keep the UI responsive.
// type: 'module' allows the worker to use ES module imports.
const mlWorker = new Worker('/src/workers/modelTrainingWorker.js', {
  type: 'module'
})

// --- Controller wiring ---
// WorkerController bridges the Web Worker ↔ DOM event bus.
// It translates postMessage/onmessage into CustomEvents that other controllers can listen to.
const w = WorkerController.init({
  worker: mlWorker,
  events: Events
})

// Try to restore a previously trained model from the backend.
// If none exists (first run or after a reseed), fall back to training from scratch.
const hasSavedModel = await w.tryLoadSavedModel()
if (!hasSavedModel) {
  const users = await userService.getDefaultUsers()
  const products = await productService.getProducts()
  w.triggerTrain(users, products)
}

// ModelController manages the "Train Model" / "Run Recommendation" buttons
ModelController.init({
  modelView,
  userService,
  productService,
  events: Events
})

// TFVisorController feeds training logs (loss, accuracy) to the tfvis dashboard
TFVisorController.init({
  tfVisorView,
  events: Events
})

// ProductController renders the product catalog and handles "Buy Now" clicks
ProductController.init({
  productView,
  userService,
  productService,
  events: Events
})

// UserController manages the user dropdown, profile details, and purchase history
const userController = UserController.init({
  userView,
  userService,
  productService,
  events: Events
})

// Add a "non-trained" user (not part of the original training data)
// so you can test recommendations for someone the model has never seen.
userController.renderUsers({
  id: 99,
  name: 'Josézin da Silva',
  age: 30,
  purchases: []
})
