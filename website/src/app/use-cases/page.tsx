"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import AnimatedSection from "@/components/AnimatedSection";
import {
  TrendingUp,
  Shield,
  BarChart3,
  ArrowRight,
  CheckCircle2,
  Building2,
  Users,
  Briefcase,
} from "lucide-react";

const useCases = [
  {
    icon: TrendingUp,
    title: "Operations",
    tagline: "MAKE MONEY",
    color: "from-blue-500 to-blue-600",
    description:
      "Design, test, and execute intelligent trading strategies that adapt to changing market conditions through continuous multimodal learning.",
    benefits: [
      "AI-driven price forecasting across 1,000+ equities",
      "Pattern recognition from historical and real-time data",
      "Event-based logic triggered by news sentiment shifts",
      "Automated execution with customizable signal frequency",
      "Backtesting with years of historical multi-frequency data",
    ],
    scenario: {
      title: "Example: Earnings Season Alpha",
      desc: 'A portfolio manager creates a Stratify agent focused on tech equities during earnings season. The agent fuses OHLCV price action with real-time earnings call summaries and analyst reports. Within seconds of an earnings release, the model generates a confidence-weighted signal — helping the manager act before the broader market digests the news.',
    },
  },
  {
    icon: Shield,
    title: "Risk Management",
    tagline: "PROTECT MONEY",
    color: "from-purple-500 to-purple-600",
    description:
      "Control exposure and ensure strategy robustness with AI-powered risk monitoring that understands both quantitative and qualitative risk factors.",
    benefits: [
      "Dynamic position sizing based on model confidence levels",
      "Intelligent stop-loss control adapting to market volatility",
      "Real-time volatility management across portfolios",
      "Stress testing against historical market crises",
      "Confidence intervals on every prediction",
    ],
    scenario: {
      title: "Example: Black Swan Detection",
      desc: "A risk officer sets up agents monitoring geopolitical news alongside VIX and cross-asset correlations. When the model detects abnormal correlation breakdowns coinciding with negative sentiment spikes, it triggers preemptive risk reduction — reducing exposure before traditional risk metrics signal danger.",
    },
  },
  {
    icon: BarChart3,
    title: "Finance",
    tagline: "TRACK & MANAGE MONEY",
    color: "from-teal-500 to-teal-600",
    description:
      "Measure performance and allocate capital intelligently with comprehensive analytics that connect strategy performance to market conditions.",
    benefits: [
      "Real-time PnL tracking with attribution analysis",
      "Comprehensive backtesting with multi-frequency data",
      "Performance metrics: Sharpe, Sortino, max drawdown",
      "Capital allocation optimization across strategies",
      "Historical signal accuracy tracking and model drift detection",
    ],
    scenario: {
      title: "Example: Portfolio Rebalancing",
      desc: "A fund allocator uses Stratify's finance module to compare agent performance across different market regimes. The system identifies which agents excel in trending vs. mean-reverting markets, enabling intelligent capital reallocation that maximizes risk-adjusted returns.",
    },
  },
];

const audiences = [
  {
    icon: Building2,
    title: "Institutional Investors",
    desc: "Hedge funds and asset managers seeking AI-augmented systematic strategies.",
  },
  {
    icon: Users,
    title: "Quantitative Teams",
    desc: "Quant researchers looking to incorporate multimodal signals into existing pipelines.",
  },
  {
    icon: Briefcase,
    title: "Independent Traders",
    desc: "Sophisticated traders who want institutional-grade tools without the infrastructure.",
  },
];

export default function UseCasesPage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid opacity-40" />
        <div className="absolute top-1/3 left-1/3 w-[600px] h-[600px] rounded-full bg-teal/10 blur-[120px]" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-24 pb-16 lg:pt-32 lg:pb-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-3xl"
          >
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium border border-teal/30 text-teal bg-teal/5 mb-6">
              <TrendingUp className="h-3 w-3" />
              Applications
            </span>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.1]">
              Real-World{" "}
              <span className="gradient-text">Use Cases</span>
            </h1>
            <p className="mt-6 text-lg text-muted max-w-2xl leading-relaxed">
              Three specialized applications powered by our unified multimodal
              intelligence layer — each designed to address a critical pillar of
              systematic trading.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Use Case Deep-Dives */}
      <section className="py-12">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 space-y-16">
          {useCases.map((uc, i) => (
            <AnimatedSection key={uc.title}>
              <div className="glass-light rounded-3xl overflow-hidden">
                <div className="p-8 sm:p-10 lg:p-12">
                  {/* Header */}
                  <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-8">
                    <div
                      className={`flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br ${uc.color} flex-shrink-0`}
                    >
                      <uc.icon className="h-7 w-7 text-white" />
                    </div>
                    <div>
                      <span className="text-[10px] font-bold tracking-widest text-muted uppercase">
                        {uc.tagline}
                      </span>
                      <h2 className="text-2xl sm:text-3xl font-bold text-white">
                        {uc.title}
                      </h2>
                    </div>
                  </div>

                  <p className="text-muted leading-relaxed mb-8 max-w-3xl">
                    {uc.description}
                  </p>

                  <div className="grid lg:grid-cols-2 gap-8">
                    {/* Benefits */}
                    <div>
                      <h4 className="text-sm font-semibold text-white mb-4">
                        Key Capabilities
                      </h4>
                      <ul className="space-y-3">
                        {uc.benefits.map((b) => (
                          <li key={b} className="flex items-start gap-3">
                            <CheckCircle2 className="h-4 w-4 text-primary-light mt-0.5 flex-shrink-0" />
                            <span className="text-sm text-muted leading-relaxed">{b}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Scenario */}
                    <div className="glass rounded-2xl p-6">
                      <h4 className="text-sm font-semibold text-primary-light mb-3">
                        {uc.scenario.title}
                      </h4>
                      <p className="text-sm text-muted leading-relaxed">
                        {uc.scenario.desc}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </AnimatedSection>
          ))}
        </div>
      </section>

      {/* Who is it for */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-background via-surface to-background" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Who Is Stratify <span className="gradient-text">For?</span>
            </h2>
          </AnimatedSection>
          <div className="grid sm:grid-cols-3 gap-6">
            {audiences.map((a, i) => (
              <AnimatedSection key={a.title} delay={i * 0.1}>
                <div className="glass-light rounded-2xl p-6 h-full text-center hover:border-primary/30 transition-colors">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 mx-auto mb-4">
                    <a.icon className="h-5 w-5 text-primary-light" />
                  </div>
                  <h4 className="text-base font-semibold text-white mb-2">{a.title}</h4>
                  <p className="text-sm text-muted leading-relaxed">{a.desc}</p>
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
              <div className="absolute inset-0 bg-gradient-to-br from-teal/15 via-primary/10 to-accent/10" />
              <div className="relative glass-light rounded-3xl px-8 py-16 sm:px-16 text-center">
                <h2 className="text-3xl font-bold text-white mb-4">
                  See It In Action
                </h2>
                <p className="text-muted max-w-xl mx-auto mb-8">
                  Ready to explore how Stratify can transform your trading
                  workflow? Start with a free trial or schedule a demo.
                </p>
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                  <Link
                    href="/signup"
                    className="group flex items-center gap-2 px-8 py-3.5 text-sm font-semibold text-white rounded-xl bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
                  >
                    Start Free Trial
                    <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition-transform" />
                  </Link>
                  <Link
                    href="/how-it-works"
                    className="px-8 py-3.5 text-sm font-medium text-white rounded-xl border border-border hover:bg-white/5 transition-colors"
                  >
                    How It Works
                  </Link>
                </div>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>
    </>
  );
}
