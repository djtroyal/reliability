import { toPng } from 'html-to-image'
import { jsPDF } from 'jspdf'

/**
 * Export an analysis-results DOM node to a multi-page A4 portrait PDF (#19).
 * Rasterizes the element to PNG, then slices it across pages if it's taller
 * than a single page (standard html2canvas->jsPDF multipage pattern).
 */
export async function exportResultsToPdf(
  element: HTMLElement | null, baseName = 'results',
): Promise<void> {
  if (!element) throw new Error('Nothing to export.')

  const png = await toPng(element, {
    backgroundColor: '#ffffff', pixelRatio: 2, cacheBust: true,
  })

  // Load the rasterized PNG to get its natural pixel dimensions.
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const i = new Image()
    i.onload = () => resolve(i)
    i.onerror = reject
    i.src = png
  })

  const pdf = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' })
  const pageWidth = pdf.internal.pageSize.getWidth()
  const pageHeight = pdf.internal.pageSize.getHeight()

  // Scale the image to the page width; total height in PDF points.
  const imgWidth = pageWidth
  const imgHeight = (img.height * imgWidth) / img.width

  let heightLeft = imgHeight
  let position = 0

  pdf.addImage(png, 'PNG', 0, position, imgWidth, imgHeight)
  heightLeft -= pageHeight

  while (heightLeft > 0) {
    position -= pageHeight
    pdf.addPage()
    pdf.addImage(png, 'PNG', 0, position, imgWidth, imgHeight)
    heightLeft -= pageHeight
  }

  pdf.save(`${baseName}.pdf`)
}
