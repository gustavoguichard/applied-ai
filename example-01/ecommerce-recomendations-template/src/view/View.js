/**
 * Base View — provides shared template utilities for all views.
 *
 * This is a simple template engine:
 * 1. loadTemplate(path) — fetches an .html file from the server as raw text
 * 2. replaceTemplate(template, data) — replaces {{key}} placeholders with values
 *
 * Concrete views (ProductView, UserView, etc.) extend this class
 * and use these methods to render their HTML snippets from /src/view/templates/.
 */
export class View {
  constructor() {
    this.loadTemplate = this.loadTemplate.bind(this)
  }

  // Fetches an HTML template file as a string (e.g. './src/view/templates/product-card.html')
  async loadTemplate(templatePath) {
    const response = await fetch(templatePath)
    return await response.text()
  }

  // Replaces all {{key}} placeholders in the template string with values from the data object.
  // Example: replaceTemplate('<p>{{name}}</p>', { name: 'Ana' }) → '<p>Ana</p>'
  replaceTemplate(template, data) {
    let result = template
    for (const [key, value] of Object.entries(data)) {
      result = result.replace(new RegExp(`{{${key}}}`, 'g'), value)
    }
    return result
  }
}
