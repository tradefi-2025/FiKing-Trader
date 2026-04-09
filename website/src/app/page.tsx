"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import AnimatedSection from "@/components/AnimatedSection";
import {
  BarChart3,
  Brain,
  Shield,
  TrendingUp,
  Zap,
  LineChart,
  Newspaper,
  Target,
  ArrowRight,
  ChevronRight,
} from "lucide-react";

/* ─── Hero ─── */
function Hero() {
  return (
    <section className="relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 bg-grid opacity-40" />
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] rounded-full bg-primary/10 blur-[120px]" />
      <div className="absolute bottom-0 right-0 w-[500px] h-[500px] rounded-full bg-accent/10 blur-[100px]" />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-24 pb-20 lg:pt-36 lg:pb-32">
        <div className="text-center max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium border border-primary/30 text-primary-light bg-primary/5 mb-6">
              <Zap className="h-3 w-3" />
              Multimodal Intelligence for Trading
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl sm:text-5xl lg:text-7xl font-bold tracking-tight text-white leading-[1.1]"
          >
            Trading Intelligence{" "}
            <span className="gradient-text">From Multimodal</span>{" "}
            Understanding
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-6 text-lg sm:text-xl text-muted max-w-2xl mx-auto leading-relaxed"
          >
            We blend years of market data with news signals through a large
            multimodal model. The result: intelligent strategies, controlled
            risk, and measurable performance.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link
              href="/signup"
              className="group flex items-center gap-2 px-8 py-3.5 text-sm font-semibold text-white rounded-xl bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
            >
              Start Free Trial
              <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition-transform" />
            </Link>
            <Link
              href="/how-it-works"
              className="flex items-center gap-2 px-8 py-3.5 text-sm font-medium text-white rounded-xl border border-border hover:bg-white/5 transition-colors"
            >
              See How It Works
            </Link>
          </motion.div>
        </div>

        {/* Data flow badges */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mt-20 flex flex-wrap items-center justify-center gap-4"
        >
          {[
            { icon: LineChart, label: "Time Series", sub: "Years of Data" },
            { icon: Newspaper, label: "News & Events", sub: "Market Context" },
            { icon: Target, label: "Signals", sub: "Alternative Data" },
            { icon: Brain, label: "Multimodal AI", sub: "Pattern Recognition" },
          ].map(({ icon: Icon, label, sub }) => (
            <div
              key={label}
              className="glass-light rounded-xl px-5 py-3.5 flex items-center gap-3 hover:border-primary/30 transition-colors"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-accent/20">
                <Icon className="h-5 w-5 text-primary-light" />
              </div>
              <div>
                <p className="text-sm font-medium text-white">{label}</p>
                <p className="text-xs text-muted">{sub}</p>
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

/* ─── How It Works (brief) ─── */
function HowItWorksBrief() {
  const steps = [
    {
      num: "01",
      title: "Data Fusion",
      desc: "Years of time series data combined with news and social signals across multiple stock entities.",
      icon: "🔀",
    },
    {
      num: "02",
      title: "Multimodal Learning",
      desc: "Large multimodal model learns patterns, relationships, and predictive signals from fused data.",
      icon: "🧠",
    },
    {
      num: "03",
      title: "Downstream Tasks",
      desc: "Three specialized applications: Operations, Risk Management, and Finance intelligence.",
      icon: "🎯",
    },
  ];

  return (
    <section className="py-24 relative">
      <div className="absolute inset-0 bg-gradient-to-b from-background via-surface to-background" />
      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <AnimatedSection className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-white">
            How It <span className="gradient-text">Works</span>
          </h2>
          <p className="mt-4 text-muted max-w-xl mx-auto">
            From raw data to intelligent decisions — in three steps.
          </p>
        </AnimatedSection>

        <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
          {steps.map((s, i) => (
            <AnimatedSection key={s.num} delay={i * 0.15}>
              <div className="glass-light rounded-2xl p-8 h-full hover:border-primary/30 transition-colors group">
                <div className="flex items-center gap-3 mb-5">
                  <span className="text-3xl">{s.icon}</span>
                  <span className="text-xs font-mono text-primary-light bg-primary/10 px-2 py-0.5 rounded-md">
                    STEP {s.num}
                  </span>
                </div>
                <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-primary-light transition-colors">
                  {s.title}
                </h3>
                <p className="text-sm text-muted leading-relaxed">{s.desc}</p>
              </div>
            </AnimatedSection>
          ))}
        </div>

        <AnimatedSection delay={0.4} className="text-center mt-10">
          <Link
            href="/how-it-works"
            className="inline-flex items-center gap-1 text-sm text-primary-light hover:text-white transition-colors"
          >
            Learn more about our approach <ChevronRight className="h-4 w-4" />
          </Link>
        </AnimatedSection>
      </div>
    </section>
  );
}

/* ─── Three Pillars ─── */
function ThreePillars() {
  const pillars = [
    {
      icon: TrendingUp,
      title: "Operations",
      tagline: "MAKE MONEY",
      desc: "Design, test, and execute intelligent trading strategies powered by AI-driven signals.",
      features: [
        "Price Forecasting",
        "Pattern Recognition",
        "Event-Based Logic",
        "Automated Execution",
      ],
      color: "from-blue-500 to-blue-600",
    },
    {
      icon: Shield,
      title: "Risk Management",
      tagline: "PROTECT MONEY",
      desc: "Control exposure and ensure strategy robustness with real-time risk monitoring.",
      features: [
        "Position Sizing",
        "Stop-Loss Control",
        "Volatility Management",
        "Stress Testing",
      ],
      color: "from-purple-500 to-purple-600",
    },
    {
      icon: BarChart3,
      title: "Finance",
      tagline: "TRACK & MANAGE MONEY",
      desc: "Measure performance and allocate capital intelligently with comprehensive analytics.",
      features: [
        "PnL Tracking",
        "Backtesting Results",
        "Performance Metrics",
        "Capital Allocation",
      ],
      color: "from-teal-500 to-teal-600",
    },
  ];

  return (
    <section className="py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <AnimatedSection className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-white">
            Three Pillars of{" "}
            <span className="gradient-text">Intelligence</span>
          </h2>
          <p className="mt-4 text-muted max-w-xl mx-auto">
            Our multimodal model drives three specialized downstream
            capabilities.
          </p>
        </AnimatedSection>

        <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
          {pillars.map((p, i) => (
            <AnimatedSection key={p.title} delay={i * 0.15}>
              <div className="glass-light rounded-2xl p-8 h-full flex flex-col hover:border-primary/30 transition-all group">
                <div
                  className={`flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${p.color} mb-5`}
                >
                  <p.icon className="h-6 w-6 text-white" />
                </div>
                <span className="text-[10px] font-bold tracking-widest text-muted uppercase mb-2">
                  {p.tagline}
                </span>
                <h3 className="text-xl font-semibold text-white mb-3">
                  {p.title}
                </h3>
                <p className="text-sm text-muted leading-relaxed mb-6">
                  {p.desc}
                </p>
                <div className="mt-auto grid grid-cols-2 gap-2">
                  {p.features.map((f) => (
                    <span
                      key={f}
                      className="text-xs text-muted bg-white/5 px-3 py-1.5 rounded-lg text-center"
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            </AnimatedSection>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ─── CTA ─── */
function CTA() {
  return (
    <section className="py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <AnimatedSection>
          <div className="relative rounded-3xl overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-accent/10 to-teal/10" />
            <div className="absolute inset-0 bg-grid opacity-30" />
            <div className="relative glass-light rounded-3xl px-8 py-16 sm:px-16 text-center">
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
                Ready to Transform Your Trading?
              </h2>
              <p className="text-muted max-w-xl mx-auto mb-8">
                Join the next generation of systematic trading. Our prototype is
                launching soon — be among the first to experience multimodal
                intelligence.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link
                  href="/signup"
                  className="group flex items-center gap-2 px-8 py-3.5 text-sm font-semibold text-white rounded-xl bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
                >
                  Get Early Access
                  <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition-transform" />
                </Link>
                <Link
                  href="mailto:admin@stratify.finance"
                  className="px-8 py-3.5 text-sm font-medium text-white rounded-xl border border-border hover:bg-white/5 transition-colors"
                >
                  Contact Us
                </Link>
              </div>
            </div>
          </div>
        </AnimatedSection>
      </div>
    </section>
  );
}

/* ─── Page ─── */
export default function Home() {
  return (
    <>
      <Hero />
      <HowItWorksBrief />
      <ThreePillars />
      <CTA />
    </>
  );
}
