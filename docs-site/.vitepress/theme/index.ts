import DefaultTheme from 'vitepress/theme'
import './style.css'

import { Button } from '../components/ui/button'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { HeroShowcase, HardwareCloud, TerminalShowcase, DemoShowcase, FeatureBento, MetricsSection, CTASection } from '../components/landing'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Button', Button)
    app.component('Card', Card)
    app.component('CardHeader', CardHeader)
    app.component('CardTitle', CardTitle)
    app.component('CardDescription', CardDescription)
    app.component('CardContent', CardContent)
    app.component('Badge', Badge)
    app.component('HeroShowcase', HeroShowcase)
    app.component('HardwareCloud', HardwareCloud)
    app.component('TerminalShowcase', TerminalShowcase)
    app.component('DemoShowcase', DemoShowcase)
    app.component('FeatureBento', FeatureBento)
    app.component('MetricsSection', MetricsSection)
    app.component('CTASection', CTASection)
  }
}
