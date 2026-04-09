"use client";

import { motion } from "framer-motion";
import AnimatedSection from "@/components/AnimatedSection";
import {
  Database,
  Newspaper,
  Brain,
  Cpu,
  LineChart,
  Shield,
  BarChart3,
  Zap,
  ArrowDown,
  Server,
  GitBranch,
} from "lucide-react";

const pipeline = [
  {
    step: "01",
    icon: Database,
    title: "Data Ingestion",
    desc: "Years of historical OHLCV time-series data is ingested from premium data providers (Refinitiv). We support multiple frequencies — from 1-minute to weekly intervals across 1,000+ equities.",
    detail: "MongoDB stores normalized time-series; PostgreSQL manages user data and agent configurations.",
    color: "from-blue-500 to-blue-600",
  },
  {
    step: "02",
    icon: Newspaper,
    title: "News & Context Collection",
    desc: "Financial news articles are collected in real-time from multiple sources including Finnhub. Articles are deduplicated and associated with specific equities and time windows.",
    detail: "Custom news retrieval prompts allow users to focus on sector-specific events or topics.",
    color: "from-indigo-500 to-indigo-600",
  },
  {
    step: "03",
    icon: Brain,
    title: "Multimodal Encoding",
    desc: "Time-series data is encoded using Kronos — a foundation model producing 512-dimensional embeddings. Financial text is encoded using FLANG-BERT, generating 768-dimensional embeddings.",
    detail: "Both encoders are fine-tuned for financial domains, capturing market-specific patterns and language.",
    color: "from-purple-500 to-purple-600",
  },
  {
    step: "04",
    icon: GitBranch,
    title: "Contextualization",
    desc: "The Naive Contextualizer fuses time-series and text embeddings into a unified 1,280-dimensional representation. This multimodal vector captures both quantitative patterns and qualitative context.",
    detail: "If only one modality is available, the system gracefully adapts its embedding strategy.",
    color: "from-violet-500 to-violet-600",
  },
  {
    step: "05",
    icon: Cpu,
    title: "Model Training",
    desc: "Users define agent parameters — equity, time horizon, features, and risk levels. The system trains a specialized model using sliding-window sequences for robust pattern recognition.",
    detail: "Training runs asynchronously via RabbitMQ workers. Model weights are stored in MongoDB.",
    color: "from-pink-500 to-pink-600",
  },
  {
    step: "06",
    icon: Zap,
    title: "Signal Generation",
    desc: "Once deployed, agents run continuously — generating trading signals every 60 seconds. Each signal includes a prediction, confidence interval, and risk assessment.",
    detail: "Inference requests use an RPC pattern over RabbitMQ for reliable, low-latency responses.",
    color: "from-teal-500 to-teal-600",
  },
];

const architecture = [
  { icon: Server, label: "API Gateway", desc: "Flask REST server handling all client requests" },
  { icon: GitBranch, label: "Message Queue", desc: "RabbitMQ for async job processing and RPC" },
  { icon: Cpu, label: "Worker Services", desc: "Distributed training & inference engines" },
  { icon: Database, label: "Data Layer", desc: "MongoDB + PostgreSQL + Redis" },
];

export default function HowItWorksPage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid opacity-40" />
        <div className="absolute top-1/3 right-1/4 w-[500px] h-[500px] rounded-full bg-primary/10 blur-[120px]" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-24 pb-16 lg:pt-32 lg:pb-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-3xl"
          >
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium border border-primary/30 text-primary-light bg-primary/5 mb-6">
              <Cpu className="h-3 w-3" />
              Platform Architecture
            </span>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.1]">
              How <span className="gradient-text">Stratify</span> Works
            </h1>
            <p className="mt-6 text-lg text-muted max-w-2xl leading-relaxed">
              From raw market data to intelligent trading signals — a deep dive
              into our six-stage pipeline that transforms multimodal data into
              actionable intelligence.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Pipeline Steps */}
      <section className="py-20">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          {pipeline.map((p, i) => (
            <AnimatedSection key={p.step} delay={i * 0.08}>
              <div className="relative">
                {/* Connector line */}
                {i < pipeline.length - 1 && (
                  <div className="absolute left-6 top-full w-px h-8 bg-gradient-to-b from-border to-transparent" />
                )}

                <div className="glass-light rounded-2xl p-8 mb-8 hover:border-primary/30 transition-colors group">
                  <div className="flex items-start gap-6">
                    <div
                      className={`flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${p.color} flex-shrink-0`}
                    >
                      <p.icon className="h-6 w-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <span className="text-xs font-mono text-primary-light bg-primary/10 px-2 py-0.5 rounded-md">
                          STEP {p.step}
                        </span>
                        <h3 className="text-xl font-semibold text-white group-hover:text-primary-light transition-colors">
                          {p.title}
                        </h3>
                      </div>
                      <p className="text-sm text-muted leading-relaxed mb-3">{p.desc}</p>
                      <p className="text-xs text-muted/70 leading-relaxed border-t border-border pt-3">
                        💡 {p.detail}
                      </p>
                    </div>
                  </div>
                </div>

                {i < pipeline.length - 1 && (
                  <div className="flex justify-center mb-4">
                    <ArrowDown className="h-4 w-4 text-border" />
                  </div>
                )}
              </div>
            </AnimatedSection>
          ))}
        </div>
      </section>

      {/* Architecture Overview */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-background via-surface to-background" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              System <span className="gradient-text">Architecture</span>
            </h2>
            <p className="mt-4 text-muted max-w-xl mx-auto">
              A distributed microservices architecture designed for scalability
              and reliability.
            </p>
          </AnimatedSection>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {architecture.map((a, i) => (
              <AnimatedSection key={a.label} delay={i * 0.1}>
                <div className="glass-light rounded-2xl p-6 text-center h-full hover:border-primary/30 transition-colors">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 mx-auto mb-4">
                    <a.icon className="h-5 w-5 text-primary-light" />
                  </div>
                  <h4 className="text-base font-semibold text-white mb-2">{a.label}</h4>
                  <p className="text-sm text-muted leading-relaxed">{a.desc}</p>
                </div>
              </AnimatedSection>
            ))}
          </div>

          {/* Data flow diagram */}
          <AnimatedSection delay={0.3} className="mt-14">
            <div className="glass-light rounded-2xl p-8">
              <h3 className="text-lg font-semibold text-white mb-6 text-center">
                End-to-End Data Flow
              </h3>
              <div className="flex flex-wrap items-center justify-center gap-3 text-xs">
                {[
                  "User Request",
                  "Flask API",
                  "RabbitMQ",
                  "Worker",
                  "Data Fetch",
                  "Contextualizer",
                  "Model Training",
                  "MongoDB (Weights)",
                  "Agent Deployed",
                  "Signal Generation",
                ].map((step, i, arr) => (
                  <span key={step} className="flex items-center gap-3">
                    <span className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-primary/10 to-accent/10 border border-border text-muted font-medium">
                      {step}
                    </span>
                    {i < arr.length - 1 && (
                      <span className="text-border hidden sm:inline">→</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Key capabilities */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Key <span className="gradient-text">Capabilities</span>
            </h2>
          </AnimatedSection>

          <div className="grid sm:grid-cols-3 gap-6">
            {[
              {
                icon: LineChart,
                title: "Real-Time Intelligence",
                desc: "Agents generate signals every 60 seconds, processing live market data with sub-second inference latency.",
              },
              {
                icon: Shield,
                title: "Robust Risk Controls",
                desc: "Built-in verification layer with confidence intervals, anomaly detection, and position-size recommendations.",
              },
              {
                icon: BarChart3,
                title: "No-Code Agent Creation",
                desc: "Users define equity, time horizon, and risk parameters — the system handles training, evaluation, and deployment.",
              },
            ].map((c, i) => (
              <AnimatedSection key={c.title} delay={i * 0.1}>
                <div className="glass-light rounded-2xl p-6 h-full hover:border-primary/30 transition-colors">
                  <c.icon className="h-8 w-8 text-primary-light mb-4" />
                  <h4 className="text-base font-semibold text-white mb-2">{c.title}</h4>
                  <p className="text-sm text-muted leading-relaxed">{c.desc}</p>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
