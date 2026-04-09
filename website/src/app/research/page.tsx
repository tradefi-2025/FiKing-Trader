"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import AnimatedSection from "@/components/AnimatedSection";
import {
  Brain,
  LineChart,
  FileText,
  Layers,
  ArrowRight,
  BookOpen,
  FlaskConical,
  Sparkles,
  Workflow,
} from "lucide-react";

const models = [
  {
    icon: LineChart,
    name: "Kronos",
    type: "Time-Series Foundation Model",
    dim: "512-dim embeddings",
    desc: "A transformer-based foundation model pre-trained on massive time-series corpora. Kronos encodes OHLCV (Open, High, Low, Close, Volume) data into rich 512-dimensional representations that capture temporal patterns, seasonality, and regime changes.",
    highlights: [
      "4 model sizes: mini (256d), small (512d), base (832d), large (1664d)",
      "Handles univariate & multivariate inputs natively",
      "Pre-trained on diverse financial time-series data",
      "Fallback encoder using statistical features (mean, std, trend, FFT)",
    ],
    color: "from-blue-500 to-cyan-500",
  },
  {
    icon: FileText,
    name: "FLANG-BERT",
    type: "Financial Language Model",
    dim: "768-dim embeddings",
    desc: "Built on SALT-NLP's FLANG-BERT architecture — a BERT model specifically fine-tuned on financial text corpora. It understands earnings reports, analyst notes, news headlines, and regulatory filings with domain-specific precision.",
    highlights: [
      "Financial domain pre-training (SEC filings, earnings calls, news)",
      "CLS token & mean pooling strategies",
      "Batch processing: 16 texts per batch for efficiency",
      "Optional masked-LM head for financial text generation",
    ],
    color: "from-purple-500 to-pink-500",
  },
];

const researchAreas = [
  {
    icon: Layers,
    title: "Multimodal Fusion",
    desc: "Our Naive Contextualizer concatenates time-series and text embeddings into a unified 1,280-dimensional vector. Current research explores cross-attention mechanisms for deeper interaction between modalities.",
  },
  {
    icon: FlaskConical,
    title: "Market Regime Detection",
    desc: "Using the fused embeddings, we're developing unsupervised methods to detect market regime changes — helping agents adapt strategies for trending, mean-reverting, and volatile environments.",
  },
  {
    icon: Sparkles,
    title: "Confidence Calibration",
    desc: "Research into calibrating prediction confidence so that stated 80% confidence actually means 80% accuracy. Critical for position sizing and risk management decisions.",
  },
  {
    icon: Workflow,
    title: "Agent Architecture Evolution",
    desc: "Moving beyond the current linear projection to attention-based architectures that can weigh different time horizons and news sources adaptively.",
  },
];

const papers = [
  {
    title: "Chronos: Learning the Language of Time Series",
    authors: "Ansari et al., 2024",
    venue: "Amazon Science",
    desc: "Foundation model for time-series forecasting that inspired our Kronos encoder architecture.",
  },
  {
    title: "FLANG-BERT: Financial Language Understanding",
    authors: "Shah et al., 2022",
    venue: "SALT-NLP / ACL",
    desc: "Domain-specific BERT model for financial NLP tasks — the base of our text encoding pipeline.",
  },
  {
    title: "Attention Is All You Need",
    authors: "Vaswani et al., 2017",
    venue: "NeurIPS",
    desc: "The transformer architecture that underpins both our time-series and text encoders.",
  },
];

export default function ResearchPage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid opacity-40" />
        <div className="absolute top-1/3 right-1/3 w-[500px] h-[500px] rounded-full bg-accent/10 blur-[120px]" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-24 pb-16 lg:pt-32 lg:pb-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-3xl"
          >
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium border border-accent/30 text-accent bg-accent/5 mb-6">
              <Brain className="h-3 w-3" />
              Research & Methodology
            </span>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.1]">
              The Science Behind{" "}
              <span className="gradient-text">Stratify</span>
            </h1>
            <p className="mt-6 text-lg text-muted max-w-2xl leading-relaxed">
              Our approach is grounded in cutting-edge research in multimodal
              learning, financial NLP, and time-series foundation models. Here is
              how we turn academic breakthroughs into trading intelligence.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Core Models */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Core <span className="gradient-text">Models</span>
            </h2>
            <p className="mt-4 text-muted max-w-xl mx-auto">
              Two specialized encoders form the backbone of our multimodal
              intelligence.
            </p>
          </AnimatedSection>

          <div className="space-y-8">
            {models.map((m, i) => (
              <AnimatedSection key={m.name} delay={i * 0.12}>
                <div className="glass-light rounded-3xl p-8 sm:p-10 hover:border-primary/30 transition-colors">
                  <div className="flex flex-col lg:flex-row gap-8">
                    <div className="flex-1">
                      <div className="flex items-center gap-4 mb-5">
                        <div
                          className={`flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${m.color}`}
                        >
                          <m.icon className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <h3 className="text-xl font-bold text-white">{m.name}</h3>
                          <p className="text-xs text-primary-light">{m.type}</p>
                        </div>
                        <span className="ml-auto text-xs font-mono text-muted bg-white/5 px-3 py-1 rounded-lg hidden sm:block">
                          {m.dim}
                        </span>
                      </div>
                      <p className="text-sm text-muted leading-relaxed mb-6">{m.desc}</p>
                    </div>
                    <div className="lg:w-80 flex-shrink-0">
                      <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">
                        Key Features
                      </h4>
                      <ul className="space-y-2">
                        {m.highlights.map((h) => (
                          <li
                            key={h}
                            className="flex items-start gap-2 text-xs text-muted leading-relaxed"
                          >
                            <span className="h-1.5 w-1.5 rounded-full bg-primary-light mt-1.5 flex-shrink-0" />
                            {h}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </AnimatedSection>
            ))}
          </div>

          {/* Fusion diagram */}
          <AnimatedSection delay={0.25} className="mt-10">
            <div className="glass-light rounded-2xl p-8 text-center">
              <h3 className="text-lg font-semibold text-white mb-6">
                Multimodal Fusion Architecture
              </h3>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4 text-xs">
                <span className="px-4 py-2 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-400 font-medium">
                  Kronos (512d)
                </span>
                <span className="text-muted">+</span>
                <span className="px-4 py-2 rounded-lg bg-purple-500/10 border border-purple-500/30 text-purple-400 font-medium">
                  FLANG-BERT (768d)
                </span>
                <span className="text-muted">→</span>
                <span className="px-4 py-2 rounded-lg bg-gradient-to-r from-primary/10 to-accent/10 border border-primary/30 text-primary-light font-medium">
                  Contextualized Vector (1,280d)
                </span>
                <span className="text-muted">→</span>
                <span className="px-4 py-2 rounded-lg bg-teal/10 border border-teal/30 text-teal font-medium">
                  Trading Signal
                </span>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Research Areas */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-background via-surface to-background" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Active Research <span className="gradient-text">Areas</span>
            </h2>
            <p className="mt-4 text-muted max-w-xl mx-auto">
              Where we are pushing the boundaries of multimodal trading
              intelligence.
            </p>
          </AnimatedSection>

          <div className="grid sm:grid-cols-2 gap-6">
            {researchAreas.map((r, i) => (
              <AnimatedSection key={r.title} delay={i * 0.1}>
                <div className="glass-light rounded-2xl p-6 h-full hover:border-primary/30 transition-colors">
                  <r.icon className="h-8 w-8 text-primary-light mb-4" />
                  <h4 className="text-base font-semibold text-white mb-2">{r.title}</h4>
                  <p className="text-sm text-muted leading-relaxed">{r.desc}</p>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Foundational Papers */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Foundational <span className="gradient-text">References</span>
            </h2>
            <p className="mt-4 text-muted max-w-xl mx-auto">
              The research that informs and inspires our approach.
            </p>
          </AnimatedSection>

          <div className="grid sm:grid-cols-3 gap-6">
            {papers.map((p, i) => (
              <AnimatedSection key={p.title} delay={i * 0.1}>
                <div className="glass-light rounded-2xl p-6 h-full hover:border-primary/30 transition-colors group">
                  <BookOpen className="h-6 w-6 text-primary-light mb-4" />
                  <h4 className="text-sm font-semibold text-white mb-1 group-hover:text-primary-light transition-colors">
                    {p.title}
                  </h4>
                  <p className="text-xs text-primary-light mb-2">
                    {p.authors} — {p.venue}
                  </p>
                  <p className="text-xs text-muted leading-relaxed">{p.desc}</p>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection>
            <div className="relative rounded-3xl overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-accent/15 via-primary/10 to-teal/10" />
              <div className="relative glass-light rounded-3xl px-8 py-16 sm:px-16 text-center">
                <h2 className="text-3xl font-bold text-white mb-4">
                  Interested in Our Research?
                </h2>
                <p className="text-muted max-w-xl mx-auto mb-8">
                  We are always looking for collaborators, researchers, and
                  partners who share our vision for multimodal trading
                  intelligence.
                </p>
                <Link
                  href="mailto:admin@stratify.finance"
                  className="group inline-flex items-center gap-2 px-8 py-3.5 text-sm font-semibold text-white rounded-xl bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
                >
                  Get In Touch
                  <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition-transform" />
                </Link>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>
    </>
  );
}
