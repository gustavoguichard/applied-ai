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
  #lossPoints = [] // Array of {x: epoch, y: lossValue} for the loss chart
  #accPoints = [] // Array of {x: epoch, y: accuracyValue} for the accuracy chart
  #toggleVisorBtn = document.querySelector('#toggleVisorBtn')
  constructor() {
    super()

    this.#toggleVisorBtn.addEventListener('click', () => {
      tfvis.visor().toggle()
    })
  }

  open() {
    tfvis.visor().open()
  }

  close() {
    tfvis.visor().close()
  }

  // Called when a new training cycle starts — clears all previous chart data
  resetDashboard() {
    this.#lossPoints = []
    this.#accPoints = []
  }

  // Renders a single line chart in the Training tab of the visor panel
  #renderChart(name, seriesLabel, yLabel, points) {
    tfvis.render.linechart(
      {
        name,
        tab: 'Training',
        style: { display: 'inline-block', width: '49%' }
      },
      { values: points, series: [seriesLabel] },
      {
        xLabel: 'Epoch (Training Cycles)',
        yLabel,
        height: 300
      }
    )
  }

  // Called once per training epoch with { epoch, loss, accuracy }.
  // Appends the new data point and re-renders both line charts.
  handleTrainingLog(log) {
    const { epoch, loss, accuracy } = log
    this.#lossPoints.push({ x: epoch, y: loss })
    this.#accPoints.push({ x: epoch, y: accuracy })

    // Accuracy chart — shows how well the model predicts over time
    this.#renderChart('Model Accuracy', 'accuracy', 'Accuracy (%)', this.#accPoints)

    // Loss chart — shows the training error decreasing over time
    this.#renderChart('Training Error', 'errors', 'Error Value', this.#lossPoints)
  }
}
