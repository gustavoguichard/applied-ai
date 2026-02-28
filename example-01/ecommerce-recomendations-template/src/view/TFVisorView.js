/**
 * TFVisorView — manages the TensorFlow.js Visualization (tfvis) dashboard.
 *
 * tfvis is a library that provides a side-panel visor for visualizing
 * ML training metrics in real time. This view:
 *
 * - Opens the visor panel on construction
 * - Provides a toggle button (top-right corner) to show/hide it
 * - Accumulates training log data (epoch, loss, accuracy) as {x, y} points
 * - Renders two live-updating line charts: Model Accuracy and Training Loss
 * - Resets all accumulated data when a new training cycle starts
 *
 * The `tfvis` global is loaded via a <script> tag in index.html
 * (not an npm import — it's a UMD bundle from CDN).
 */
import { View } from './View.js'

export class TFVisorView extends View {
  // State accumulated across training epochs for charting
  #weights = null
  #catalog = []
  #users = []
  #logs = []
  #lossPoints = [] // Array of {x: epoch, y: lossValue} for the loss chart
  #accPoints = [] // Array of {x: epoch, y: accuracyValue} for the accuracy chart
  #toggleVisorBtn = document.querySelector('#toggleVisorBtn')
  constructor() {
    super()

    tfvis.visor().open()
    this.#toggleVisorBtn.addEventListener('click', () => {
      tfvis.visor().toggle()
    })
  }

  renderData(data) {
    this.#weights = data.weights
    this.#catalog = data.catalog
    this.#users = data.users
  }

  // Called when a new training cycle starts — clears all previous chart data
  resetDashboard() {
    this.#weights = null
    this.#catalog = []
    this.#users = []
    this.#logs = []
    this.#lossPoints = []
    this.#accPoints = []
  }

  // Called once per training epoch with { epoch, loss, accuracy }.
  // Appends the new data point and re-renders both line charts.
  handleTrainingLog(log) {
    const { epoch, loss, accuracy } = log
    this.#lossPoints.push({ x: epoch, y: loss })
    this.#accPoints.push({ x: epoch, y: accuracy })
    this.#logs.push(log)

    // Accuracy chart — shows how well the model predicts over time
    tfvis.render.linechart(
      {
        name: 'Precisão do Modelo',
        tab: 'Treinamento',
        style: { display: 'inline-block', width: '49%' }
      },
      { values: this.#accPoints, series: ['precisão'] },
      {
        xLabel: 'Época (Ciclos de Treinamento)',
        yLabel: 'Precisão (%)',
        height: 300
      }
    )

    // Loss chart — shows the training error decreasing over time
    tfvis.render.linechart(
      {
        name: 'Erro de Treinamento',
        tab: 'Treinamento',
        style: { display: 'inline-block', width: '49%' }
      },
      { values: this.#lossPoints, series: ['erros'] },
      {
        xLabel: 'Época (Ciclos de Treinamento)',
        yLabel: 'Valor do Erro',
        height: 300
      }
    )
  }
}
