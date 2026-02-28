# E-commerce Recommendation System

A web application that displays user profiles and product listings, with the ability to track user purchases for future machine learning recommendations using TensorFlow.js.

## Setup and Run

1. Install dependencies:
```
npm install
```

2. Start the application:
```
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Features

- User profile selection with details display
- Past purchase history display (add and remove purchases)
- Product listing with "Buy Now" functionality
- Purchase tracking using sessionStorage
- TensorFlow.js model training on a Web Worker (off the main thread)
- Real-time training visualization with tfvis (accuracy and loss charts)
- ML-based product recommendations (currently a stub — training and inference logic to be implemented)

## Architecture Overview

This is a **vanilla JavaScript SPA** (no framework) that follows an **MVC-like architecture** with **event-driven communication** between components. There is no build step — it uses native ES modules (`type: "module"`) served directly via `browser-sync`.

```
index.html (entry point)
  └── src/index.js (bootstrapper)
        ├── Services (data layer)
        │     ├── UserService    → sessionStorage + fetch
        │     └── ProductService → fetch
        ├── Views (DOM layer)
        │     ├── View           → base class (template loading)
        │     ├── UserView
        │     ├── ProductView
        │     ├── ModelTrainingView
        │     └── TFVisorView
        ├── Controllers (glue layer)
        │     ├── UserController
        │     ├── ProductController
        │     ├── ModelTrainingController
        │     ├── TFVisorController
        │     └── WorkerController
        ├── Events (pub/sub bus)
        │     └── Events class → CustomEvent on `document`
        └── Workers (off-thread ML)
              └── modelTrainingWorker.js → Web Worker
```

## Project Structure

```
├── index.html              → Main HTML (Bootstrap layout, script tags)
├── style.css               → Application styles
├── data/
│   ├── products.json       → Product catalog (10 items)
│   └── users.json          → Default users with purchase history (5 users)
└── src/
    ├── index.js             → Bootstrapper: wires services, views, controllers
    ├── events/
    │   ├── constants.js     → Event name constants (DOM + Worker namespaces)
    │   └── events.js        → Event bus: pub/sub via DOM CustomEvents
    ├── service/
    │   ├── UserService.js   → User CRUD over sessionStorage
    │   └── ProductService.js → Read-only product catalog (cached after first fetch)
    ├── view/
    │   ├── View.js          → Base class: template loading + {{mustache}} replacement
    │   ├── UserView.js      → User dropdown, age, past purchases list
    │   ├── ProductView.js   → Product cards grid, "Buy Now" buttons
    │   ├── ModelTrainingView.js → Train/Recommend buttons, progress spinner
    │   └── TFVisorView.js   → tfvis dashboard (accuracy + loss charts)
    ├── controller/
    │   ├── UserController.js          → User selection, purchase add/remove
    │   ├── ProductController.js       → Product rendering, buy flow
    │   ├── ModelTrainingController.js → Training state machine, recommend flow
    │   ├── TFVisorController.js       → Feeds training logs to visor charts
    │   └── WorkerController.js        → Bridges Web Worker ↔ DOM event bus
    └── workers/
        └── modelTrainingWorker.js     → Web Worker for TF.js training (stub)
```

## Layer-by-Layer Breakdown

### Data Layer — `data/` + Services

**Static data** lives in JSON files:
- `data/users.json` — 5 users, each with `id`, `name`, `age`, and `purchases` (array of product IDs, e.g. `[1, 2]`)
- `data/products.json` — 10 products with `id`, `name`, `category`, `price`, `color`

Purchases are **normalized** — users store only product IDs, not full product objects. Controllers resolve IDs to full product objects via `ProductService` when needed for rendering.

**`UserService`** wraps `sessionStorage` as a mutable in-memory DB. On first call to `getDefaultUsers()`, it fetches `users.json` and caches everything in `sessionStorage` under the key `ew-academy-users`. All subsequent reads/writes (get, update, add) go through `sessionStorage`, meaning the data is ephemeral per browser tab.

**`ProductService`** is read-only — fetches `products.json` once and caches the result in memory for subsequent lookups.

### View Layer — `View` base + concrete views

**`View`** (base class) provides two utilities:
- `loadTemplate(path)` — fetches an HTML file as text
- `replaceTemplate(template, data)` — Mustache-style `{{key}}` string replacement

Each concrete view grabs DOM elements via `document.querySelector` in class field initializers, loads its HTML templates at construction time, and exposes callback registration methods (Inversion of Control — the controller tells the view what to do on user interaction).

- **`UserView`** — manages the user dropdown, age field, and past purchases list. Clicking a purchase removes it (fade-out animation).
- **`ProductView`** — renders product cards from template. Buy buttons are disabled until a user is selected. Clicking "Buy Now" triggers a brief green "Added" animation.
- **`ModelView`** — manages "Train Model" / "Run Recommendation" buttons and the collapsible "All Users Purchase Data" section. Shows a spinner during training.
- **`TFVisorView`** — wraps TensorFlow.js's `tfvis` visor. Renders live accuracy and loss line charts during training epochs.

### Event Bus — `Events` class

This is the **central nervous system** of the app. It's a static class that wraps `document.addEventListener` / `document.dispatchEvent` with `CustomEvent`. No instances — every controller imports it and calls static methods.

Key events (from `constants.js`):

| Event | Dispatched by | Listened by |
|---|---|---|
| `user:selected` | UserController | ProductController, ModelController |
| `users:updated` | UserController | ModelController |
| `purchase:added` | ProductController | UserController |
| `purchase:remove` | UserView (via controller) | UserController |
| `training:train` | ModelController | WorkerController, TFVisorController |
| `training:complete` | WorkerController | ModelController, WorkerController |
| `model:progress-update` | WorkerController | ModelController |
| `recommend` | ProductController, ModelController | WorkerController |
| `recommendations:ready` | WorkerController | ProductController |
| `tfvis:logs` | WorkerController | TFVisorController |

### Controller Layer — wiring it all together

Each controller follows the same pattern:
1. Receives dependencies via `static init(deps)` (factory method)
2. Registers callbacks on its view (what to do when user clicks)
3. Subscribes to events from other controllers
4. Dispatches events to communicate outward

**`UserController`** — the user lifecycle:
- Renders user dropdown options (5 from JSON + 1 hardcoded "Josézin da Silva")
- On user select → fetches user from service → dispatches `user:selected` → resolves purchase IDs via ProductService → renders details + purchases
- On purchase added (from ProductController) → pushes product ID to user's purchases → updates sessionStorage → renders new purchase → dispatches `users:updated`
- On purchase removed (click on past-purchase) → removes ID from array → updates storage → dispatches `users:updated`

**`ProductController`** — the product catalog:
- Fetches all products → renders cards (disabled buttons)
- On `user:selected` → enables buy buttons + dispatches `recommend` (to get ML recommendations)
- On `recommendations:ready` → re-renders the product list with recommended products
- On "Buy Now" click → dispatches `purchase:added`

**`ModelTrainingController`** — the ML orchestrator:
- On "Train Model" click → gets all users → dispatches `training:train`
- On `training:complete` → sets `#alreadyTrained = true`, enables "Run Recommendation" button (if user is selected)
- On "Run Recommendation" click → dispatches `recommend` for current user
- On `users:updated` → resolves purchase IDs to products via ProductService → refreshes the "All Users Purchase Data" table

**`WorkerController`** — the bridge to the Web Worker:
- Listens to `training:train` → `postMessage` to worker with `train:model` action
- Listens to `recommend` → `postMessage` to worker (only if already trained)
- Worker's `onmessage` → translates worker events back into DOM `CustomEvent`s (`progress:update` → `model:progress-update`, `training:complete` → `training:complete`, etc.)

**`TFVisorController`** — visualization:
- On `training:train` → resets the visor dashboard
- On `tfvis:logs` → feeds epoch/loss/accuracy data to TFVisorView for charting

### Web Worker — `modelTrainingWorker.js`

Runs on a separate thread. Currently a **stub/template** — it imports TensorFlow.js but doesn't build a real model yet. It:
- Receives `train:model` → posts fake 50% progress, then after 1s posts 100% + `training:complete`
- Receives `recommend` → logs to console but the actual `postMessage` with recommendations is commented out

The `_globalCtx` object is there to eventually hold the trained model state.

## Startup Flow

```
1. Create services (UserService, ProductService)
2. Create views (UserView, ProductView, ModelView, TFVisorView)
3. Create Web Worker for ML training
4. WorkerController.init() — wires worker ↔ events bridge
5. Fetch default users → immediately trigger training on the worker
6. ModelController.init() — wires train/recommend buttons + events
7. TFVisorController.init() — wires visor dashboard
8. ProductController.init() → fetches products → renders catalog
9. UserController.init() → fetches users + adds "Josézin da Silva" (id: 99)
   → renders dropdown → dispatches users:updated
```

On page load, the model auto-trains with the 5 default users. The tfvis visor opens and shows the (stub) training charts. Once training "completes" (1s timeout), selecting a user and clicking "Run Recommendation" will attempt recommendations (though the worker's `recommend` function doesn't return results yet).

## Key Design Decisions

1. **Event-driven decoupling** — controllers never reference each other directly. All cross-controller communication goes through `Events`, making each controller independently testable.

2. **Normalized purchases** — users store only product IDs in their `purchases` array, not full product objects. Controllers resolve IDs to products via `ProductService` at the boundary before passing data to views. This keeps a single source of truth for product data.

3. **sessionStorage as database** — user mutations (adding/removing purchases) persist only within the tab session. Refreshing reloads defaults from JSON.

4. **Web Worker for ML** — TensorFlow.js model training is off the main thread, keeping the UI responsive. The worker communicates via `postMessage`/`onmessage`.

5. **HTML template files** — views fetch `.html` snippets and do `{{mustache}}` replacement, keeping presentation separate from logic without needing a build step.

6. **No build system** — native ES modules (`type: "module"`), served directly via `browser-sync`. No bundler, no transpilation.

## Future Enhancements

- TensorFlow.js-based recommendation engine (replace the stub in `modelTrainingWorker.js`)
- User similarity analysis
- Product recommendation based on purchase history
