import { toSvg, toPng, toJpeg } from 'html-to-image'
import { jsPDF } from 'jspdf'

export type DiagramFormat = 'svg' | 'png' | 'jpg' | 'pdf'

function triggerDownload(dataUrl: string, filename: string) {
  const a = document.createElement('a')
  a.href = dataUrl
  a.download = filename
  a.click()
}

/**
 * Export a diagram DOM node (e.g. a ReactFlow wrapper) to SVG / PNG / JPG / PDF
 * (#19). Pass the element to capture and the desired format.
 */
export async function exportDiagram(
  element: HTMLElement | null, format: DiagramFormat, baseName = 'diagram',
): Promise<void> {
  if (!element) throw new Error('Nothing to export.')
  const opts = { backgroundColor: '#ffffff', pixelRatio: 2, cacheBust: true }

  if (format === 'svg') {
    const dataUrl = await toSvg(element, { backgroundColor: '#ffffff', cacheBust: true })
    triggerDownload(dataUrl, `${baseName}.svg`)
    return
  }
  if (format === 'png') {
    triggerDownload(await toPng(element, opts), `${baseName}.png`)
    return
  }
  if (format === 'jpg') {
    triggerDownload(await toJpeg(element, { ...opts, quality: 0.95 }), `${baseName}.jpg`)
    return
  }
  // PDF: rasterize to PNG, place on a page sized to the diagram.
  const png = await toPng(element, opts)
  const w = element.offsetWidth || 800
  const h = element.offsetHeight || 600
  const pdf = new jsPDF({
    orientation: w >= h ? 'landscape' : 'portrait',
    unit: 'px',
    format: [w, h],
  })
  pdf.addImage(png, 'PNG', 0, 0, w, h)
  pdf.save(`${baseName}.pdf`)
}
